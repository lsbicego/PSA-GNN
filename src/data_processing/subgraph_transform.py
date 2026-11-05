import torch
from torch_geometric.data import Data
from torch.nn import functional as F
from typing import List, Optional
from functools import partial
from omegaconf import OmegaConf

class SubgraphData(Data):
    """PyG ``Data`` object extended with explicit subgraph structure.

    Args:
        Inherits the standard ``torch_geometric.data.Data`` fields and stores
        additional subgraph tensors such as ``sg_nodes_orig``, ``sg_batch``,
        ``sg_edge_index``, and ``sg_graph``.

    Returns:
        A batched data container whose custom indexing rules keep subgraph-level
        tensors consistent across PyG collation.
    """
    def __inc__(self, key, value, *args, **kwargs):
        # Standard graph-level indices
        if key == 'edge_index':
            return self.num_nodes

        # Subgraph-level graph over subgraph-node copies
        if key == 'sg_edge_index':
            return self.num_sg_nodes

        # "batch" over original nodes (standard PyG)
        if key == 'batch':
            return int(self.batch.max()) + 1 if self.batch.numel() > 0 else 0

        # "batch" over subgraph-node copies (which subgraph)
        if key == 'sg_batch':
            # return int(self.sg_batch.max()) + 1 if self.sg_batch.numel() > 0 else 0
            return self.num_subgraphs

        # "batch" over subgraphs (which original graph)
        if key == 'sg_graph':
            # return int(self.sg_graph.max()) + 1 if self.sg_graph.numel() > 0 else 0
            return 1 # since we only handle single-graph Data here
        
        if key == 'sg_edges_orig':
            # When batching, edge_attr for each graph is concatenated,
            # so we offset by the number of edges in this graph.
            return self.edge_index.size(1)
        
        if key == 'sg_nodes_orig':
            # sg_nodes_orig indexes into x; when batching, x is concatenated,
            # so we must offset by the number of nodes in the previous graphs.
            return self.num_nodes
        
        if key == 'super_edge_index':
            return self.num_subgraphs

        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Edge indices are concatenated along dim=1, everything else along dim=0.
        if key in ['edge_index', 'sg_edge_index', "super_edge_index"]:
            return -1
        return 0

    @property
    def num_sg_nodes(self):
        return int(self.sg_nodes_orig.size(0)) if hasattr(self, 'sg_nodes_orig') else 0

    @property
    def num_subgraphs(self):
        return int(self.sg_graph.size(0)) if hasattr(self, 'sg_graph') else 0
    
    def subgraph_nodes(self, sg_id: int):
        """Original node IDs belonging to subgraph sg_id."""
        mask = (self.sg_batch == sg_id)
        return self.sg_nodes_orig[mask]
    
    def subgraph_original_edges(self, sg_id: int):
        """
        Return a 1D tensor of original edge indices (into self.edge_index)
        that belong to subgraph sg_id. Teleport edges are excluded.
        """
        assert hasattr(self, "sg_edges_orig"), "sg_edges_orig is required to recover original edges."

        Te_real = self.sg_edges_orig.size(0)
        if Te_real == 0:
            return self.sg_edges_orig.new_empty(0, dtype=torch.long)

        # Only consider real edges (first Te_real columns)
        sg_edge_index_real = self.sg_edge_index[:, :Te_real]
        src = sg_edge_index_real[0]
        dst = sg_edge_index_real[1]

        sg_src = self.sg_batch[src]
        sg_dst = self.sg_batch[dst]

        # Real subgraph edges should have both endpoints in the same subgraph
        mask = (sg_src == sg_id) & (sg_dst == sg_id)

        return self.sg_edges_orig[mask]


# register as saefe global for serialization
torch.serialization.add_safe_globals([SubgraphData])


class SubgraphTransform:
    """Convert one graph into ``SubgraphData`` using a configured policy.

    Args:
        cfg: Experiment configuration containing the subgraph policy,
            marking settings, and optional subgraph-size encoding.

    Returns:
        A callable transform that packages policy outputs into the subgraph
        tensor format used by the models in this repository.
    """
    def __init__(self, cfg):
        
        policy_config = OmegaConf.to_object(cfg.subgraph_policy)
        name = policy_config.pop("method")
        self.policy_name = name
        self.marking_level = policy_config.pop("marking_level")
        self.marking_settings = policy_config.pop("marking_settings")[self.marking_level]

        if self.marking_settings is None:
            self.marking_settings = []

        self.khop = policy_config.get("khop", 0)
        self.max_k = 5

        self.use_node_markings = cfg.subgraph_policy.use_node_markings
        self.use_edge_markings = cfg.subgraph_policy.use_edge_markings
        self.add_subgraph_size = cfg.dataset.model_params.get("encode_subgraph_size", False)

        # resolve policy function
        match name:
            case "leiden":
                from .structural_policies import leiden_khop_policy_v2
                self.policy_fn = partial(leiden_khop_policy_v2,
                            **policy_config)
            case "acl":
                from .acl_policy import acl_push_auto_policy
                self.policy_fn = partial(acl_push_auto_policy,
                            **policy_config)
            case "egonet":
                from .seed_policies import ego_khop_policy
                self.policy_fn = partial(ego_khop_policy,
                            **policy_config)
            case "seeded_hops":
                from .seed_policies import seeded_tesselating_egonet
                self.policy_fn = partial(seeded_tesselating_egonet,
                            **policy_config)
            case "whole_graph":
                from .seed_policies import whole_graph_policy
                self.policy_fn = whole_graph_policy
            case "cliquepool":
                from .structural_policies import clique_khop_policy
                self.policy_fn = partial(clique_khop_policy,
                            **policy_config)
            case "curvpool":
                from .structural_policies import curv_khop_policy
                self.policy_fn = partial(curv_khop_policy,
                            **policy_config)
            case "cliquepool_maximal":
                from .structural_policies import clique_khop_policy
                self.policy_fn = partial(clique_khop_policy,
                            **policy_config)
            case _:
                raise ValueError(f"Unknown overlapping subgraph method '{name}'.")
        
    def build_node_marking_tensor(self, marking_dict: dict) -> torch.Tensor:
        """Assemble the configured node-marking tensor for one subgraph.

        Args:
            marking_dict: Mapping from marking names to per-node tensors
                produced by a subgraph policy.

        Returns:
            A dense marking tensor with one column block per configured marking.
        """
        mark_tensors = []
        for key in self.marking_settings:

            if not key in marking_dict:
                raise ValueError(f"Requested marking '{key}' not found in marking_dict.")
            mark = marking_dict[key]
            if mark is None:
                raise ValueError(f"Requested marking '{key}' is None in marking_dict.")

            # if all checks passed -> process tensor
            if key == "depth_core":
                # one hot with classes = khop
                mark = F.one_hot(mark.clamp(max=self.khop), num_classes=self.khop + 1).to(torch.float)
            elif key == "depth_root":
                # one hot with classes = 5 (max_k)
                mark = F.one_hot(mark.clamp(max=self.max_k), num_classes=self.max_k + 1).to(torch.float)
            elif key == "incident_count":
                # encode with clamped int to 10
                mark = F.one_hot(mark.clamp(max=10).long(), num_classes=11).to(torch.float)
            else: 
                mark = mark.to(torch.float)
            
            mark_tensors.append(mark.unsqueeze(-1) if mark.dim() == 1 else mark)
                        
        if len(mark_tensors) == 0 and self.use_node_markings:
            raise ValueError("No markings were found to build the marking tensor.")

        return torch.cat(mark_tensors, dim=-1)

    def finalize_policy_output(self, raw_subgraphs):
        """Normalize raw policy output into the repository's subgraph schema.

        Args:
            raw_subgraphs: List of per-subgraph dictionaries returned by a
                policy function.

        Returns:
            A dictionary of aligned node lists, edge lists, markings, and
            optional visualization/debug tensors.
        """
        node_lists: List[torch.Tensor] = []
        edge_lists: List[torch.Tensor] = []
        node_markings: Optional[List[Optional[torch.Tensor]]] = [] if self.use_node_markings else None
        edge_markings: Optional[List[Optional[torch.Tensor]]] = [] if self.use_edge_markings else None
        edge_debug: List[Optional[torch.Tensor]] = []
        
        is_root_list: Optional[List[Optional[torch.Tensor]]] = [] if self.use_node_markings else None
        is_core_list: Optional[List[Optional[torch.Tensor]]] = [] if self.use_node_markings else None
        score_list: Optional[List[Optional[torch.Tensor]]] = [] if self.use_node_markings else None

        has_roots = raw_subgraphs[0].get("is_root", None) is not None
        has_cores = raw_subgraphs[0].get("is_core", None) is not None
        has_scores = raw_subgraphs[0].get("score", None) is not None
        policy_metadata = raw_subgraphs[0].get("policy_metadata", None)

        for sg in raw_subgraphs:
            node_lists.append(sg["nodes"])
            edge_lists.append(sg["edges"])
            edge_debug.append(sg.get("edge_debug", None))

            
            if self.use_node_markings:
                # keeps chosen markings only and one-hot encodes as appropriate
                node_marking_dict = sg["node_marking_dict"]
                marking_tensor = self.build_node_marking_tensor(node_marking_dict)
                node_markings.append(marking_tensor)

                is_root_list.append(sg.get("is_root", None) if has_roots else None)
                is_core_list.append(sg.get("is_core", None) if has_cores else None)
                score_list.append(sg.get("score", None) if has_scores else None)

            if self.use_edge_markings:
                edge_markings.append(sg.get("edge_markings", None))

        return {
            "node_lists": node_lists,
            "edge_lists": edge_lists,
            "node_markings": node_markings if self.use_node_markings else None,
            "edge_markings": edge_markings if self.use_edge_markings else None,
            "is_root_list": is_root_list if has_roots else None,
            "is_core_list": is_core_list if has_cores else None,
            "score_list": score_list if has_scores else None,
            "edge_debug": edge_debug,
            "policy_metadata": policy_metadata,
        }


    def __call__(self, data: Data) -> SubgraphData:
        """Apply the configured subgraph policy and package the result.

        Args:
            data: Single input graph before subgraph preprocessing.

        Returns:
            A ``SubgraphData`` object containing original graph tensors together
            with explicit subgraph membership, subgraph edges, and supernode
            graph connectivity.
        """
        raw_subgraphs = self.policy_fn(data)

        policy_out = self.finalize_policy_output(raw_subgraphs)

        node_lists = policy_out["node_lists"]          # List[Tensor]
        edge_lists = policy_out["edge_lists"]          # List[Tensor]
        node_markings = policy_out.get("node_markings", None)
        edge_markings = policy_out.get("edge_markings", None)
        edge_debug = policy_out.get("edge_debug", None)
        policy_metadata = policy_out.get("policy_metadata", None)


        is_root_list = policy_out.get("is_root_list", None)
        is_core_list = policy_out.get("is_core_list", None)
        score_list = policy_out.get("score_list", None)

        assert len(node_lists) == len(edge_lists), "node_lists and edge_lists must have same length"
        num_subgraphs = len(node_lists)

        has_mark = node_markings is not None
        if has_mark:
            assert len(node_markings) == num_subgraphs, "node_markings length mismatch"

        has_edge_mark = edge_markings is not None
        if has_edge_mark:
            assert len(edge_markings) == num_subgraphs, "edge_markings length mismatch"

        # Infer edge-level feature dims even if a graph has zero subgraph edges.
        # This keeps collation stable across graphs with/without edges.
        edge_mark_dim = None
        if has_edge_mark:
            for em in edge_markings:
                if em is not None:
                    edge_mark_dim = int(em.size(-1))
                    break
            if edge_mark_dim is None:
                edge_mark_dim = 0

        edge_debug_dim = None
        if edge_debug is not None:
            for dbg in edge_debug:
                if dbg is not None:
                    edge_debug_dim = int(dbg.size(-1))
                    break
            if edge_debug_dim is None:
                edge_debug_dim = 0

        assert len(node_lists) > 0, f"No subgraphs generated by policy. Debugging statements: {policy_out}, \n has_mark={has_mark}, has_edge_mark={has_edge_mark}"

        N = data.num_nodes
        E = data.edge_index.size(1)

        # 1) Flatten nodes & markings across subgraphs
        total_nodes = sum(nodes.numel() for nodes in node_lists)
        device = data.edge_index.device

        sg_nodes_orig = torch.empty(total_nodes, dtype=torch.long, device=device)
        sg_batch = torch.empty(total_nodes, dtype=torch.long, device=device)

        if is_root_list:
            sg_is_root = torch.empty(total_nodes, 1, dtype=torch.long, device=device)
        if is_core_list:
            sg_is_core = torch.empty(total_nodes, 1, dtype=torch.long, device=device)
        if score_list:
            sg_score = torch.empty(total_nodes, 1, dtype=torch.float, device=device)

        if has_mark:
            mark_dim = node_markings[0].size(-1)
            if self.add_subgraph_size:
                mark_dim += 1
            sg_node_mark = torch.empty(total_nodes, mark_dim, dtype=node_markings[0].dtype,
                                       device=node_markings[0].device)
        else:
            sg_node_mark = None

        if has_edge_mark:
            sg_edge_mark_list = []

        sg_edge_debug_list = []
        sg_edge_index_list = []
        sg_edges_orig_list = []

        cursor = 0

        # Assign contiguous subgraph-node ids and local edges.
        for sg_id, (nodes_i, edges_i) in enumerate(zip(node_lists, edge_lists)):
            nodes_i = nodes_i.to(device)
            edges_i = edges_i.to(device)

            k = nodes_i.numel()
            if k == 0:
                continue

            # Record node-to-subgraph membership.
            sg_nodes_orig[cursor:cursor + k] = nodes_i
            sg_batch[cursor:cursor + k] = sg_id

            if has_mark:
                marks_i = node_markings[sg_id].to(device)
                if self.add_subgraph_size:
                    # Append subgraph size as an extra node feature.
                    size_feat = torch.full((k, 1), fill_value=k, dtype=marks_i.dtype, device=device)
                    marks_i = torch.cat([marks_i, size_feat], dim=-1)
                assert marks_i.size(0) == k, "markings must align with nodes in each subgraph"
                sg_node_mark[cursor:cursor + k] = marks_i

            # Optional visualization flags.
            if is_root_list is not None:
                root_flags = is_root_list[sg_id].to(device).view(-1, 1)
                assert root_flags.size(0) == k, "is_root flags must align with nodes in each subgraph"
                sg_is_root[cursor:cursor + k] = root_flags

            if is_core_list is not None:
                core_flags = is_core_list[sg_id].to(device).view(-1, 1)
                assert core_flags.size(0) == k, "is_core flags must align with nodes in each subgraph"
                sg_is_core[cursor:cursor + k] = core_flags

            if score_list is not None:
                scores = score_list[sg_id].to(device).view(-1, 1)
                assert scores.size(0) == k, "score values must align with nodes in each subgraph"
                sg_score[cursor:cursor + k] = scores

            # Map original node ids to local subgraph-node ids.
            mapping = {int(n.item()): cursor + j for j, n in enumerate(nodes_i)}

            # Build the local subgraph adjacency.
            if edges_i.numel() > 0:
                src = data.edge_index[0, edges_i]
                dst = data.edge_index[1, edges_i]

                # Convert original node ids to local subgraph-node ids.
                local_src = torch.tensor([mapping[int(u)] for u in src.tolist()],
                                         dtype=torch.long, device=device)
                local_dst = torch.tensor([mapping[int(v)] for v in dst.tolist()],
                                         dtype=torch.long, device=device)

                sg_edge_index_list.append(torch.stack([local_src, local_dst], dim=0))
                sg_edges_orig_list.append(edges_i)

                if has_edge_mark:
                    marks_e = edge_markings[sg_id].to(device)
                    assert marks_e.size(0) == edges_i.numel(), "edge markings must align with edges"
                    sg_edge_mark_list.append(marks_e)
                
                if edge_debug is not None:
                    dbg_e = edge_debug[sg_id]
                    if dbg_e is not None:
                        dbg_e = dbg_e.to(device)
                        assert dbg_e.size(0) == edges_i.numel(), "edge_debug must align with edges"
                        sg_edge_debug_list.append(dbg_e)

            cursor += k

        # Concatenate subgraph adjacencies and aligned original edge ids.
        if sg_edge_index_list:
            sg_edge_index = torch.cat(sg_edge_index_list, dim=1)
            sg_edges_orig = torch.cat(sg_edges_orig_list, dim=0)
        else:
            sg_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            sg_edges_orig = torch.empty(0, dtype=torch.long, device=device)

        if has_edge_mark:
            if sg_edge_mark_list:
                sg_edge_mark = torch.cat(sg_edge_mark_list, dim=0)  # [Te_total, F_edge_mark]
            else:
                sg_edge_mark = torch.empty((0, edge_mark_dim), device=device)
        else:
            sg_edge_mark = None

        # Concatenate optional edge-level debug features.
        if edge_debug is not None:
            if sg_edge_debug_list:
                sg_edge_debug = torch.cat(sg_edge_debug_list, dim=0)  # [Te_total, F_dbg]
            else:
                sg_edge_debug = torch.empty((0, edge_debug_dim), device=device)
        else:
            sg_edge_debug = None

        # Each subgraph belongs to the current graph.
        sg_graph = torch.zeros(num_subgraphs, dtype=torch.long, device=device)

        out = SubgraphData(
            x=data.x,
            y=data.y,
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
        )

        if hasattr(data, 'batch'):
            out.batch = data.batch

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            out.edge_attr = data.edge_attr
        
        if hasattr(data, 'domain'):
            out.domain = data.domain

        if hasattr(data, 'is_connected'):
            out.is_connected = data.is_connected

        if hasattr(data, 'hardness'):
            out.hardness = data.hardness

        if hasattr(data, 'category'):
            out.category = data.category

        out.sg_nodes_orig = sg_nodes_orig
        out.sg_batch = sg_batch
        out.sg_edge_index = sg_edge_index
        out.sg_graph = sg_graph

        # Optional visualization fields.
        out.sg_is_core = sg_is_core if is_core_list is not None else None
        out.sg_is_root = sg_is_root if is_root_list is not None else None
        out.sg_score = sg_score if score_list is not None else None

        if sg_node_mark is not None:
            out.sg_node_mark = sg_node_mark

        out.sg_edges_orig = sg_edges_orig

        if sg_edge_mark is not None:
            out.sg_edge_mark = sg_edge_mark

        if sg_edge_debug is not None:
            out.sg_edge_debug = sg_edge_debug

        if policy_metadata is not None:
            for key, value in policy_metadata.items():
                if torch.is_tensor(value):
                    setattr(out, key, value.to(device))
        
        out.super_edge_index, out.super_edge_weight = build_supernode_graph_scheme1(out)

        return out


def build_supernode_graph_scheme1(data):
    """Build the contracted supernode graph for full-coverage subgraphs.

    Args:
        data: ``SubgraphData`` object with original edges and subgraph
            memberships.

    Returns:
        A tuple ``(edge_index_sub, edge_weight_sub)`` describing the subgraph
        interaction graph, where weights count crossing original edges.
    """

    # ---- Step 1: Map original nodes → canonical subgraph id ----
    # For overlapping nodes, use FIRST occurrence as canonical.
    N = data.num_nodes
    P = data.num_subgraphs
    device = data.sg_batch.device

    # initialize with -1
    global_sub_id = torch.full((N,), -1, dtype=torch.long, device=device)

    # fill via first occurrence
    # data.sg_nodes_orig[i] is original node index of subgraph-copy i
    # data.sg_batch[i]       is subgraph-id of that copy
    global_sub_id[data.sg_nodes_orig] = data.sg_batch
    # If overlaps exist, first occurrence wins automatically.

    # Non-sampled policies are expected to cover all original nodes.
    uncovered = torch.nonzero(global_sub_id < 0, as_tuple=False).view(-1)
    if uncovered.numel() > 0:
        preview = uncovered[:10].tolist()
        # raise ValueError(
        #     "build_supernode_graph_scheme1 requires full node coverage: "
        #     f"{int(uncovered.numel())}/{int(N)} nodes are not assigned to any subgraph. "
        #     f"First uncovered node ids: {preview}"
        # )
        return build_supernode_graph_sampled_safe(data)

    # ---- Step 2: Lift original edge_index to subgraph graph ----
    edge_index = data.edge_index  # original edges

    su = global_sub_id[edge_index[0]]   # [E]
    sv = global_sub_id[edge_index[1]]   # [E]

    # Mask out self-loops at supernode level
    mask = su != sv
    su = su[mask]
    sv = sv[mask]

    if su.numel() == 0:
        return (torch.empty(2, 0, dtype=torch.long, device=device),
                torch.empty(0, device=device))

    # Make undirected (optional)
    row = torch.cat([su, sv], dim=0)
    col = torch.cat([sv, su], dim=0)

    # ---- Step 3: Coalesce into weighted patch graph ----
    from torch_geometric.utils import coalesce, remove_self_loops

    edge_index_sub, edge_weight_sub = coalesce(
        torch.stack([row, col], dim=0),
        torch.ones(row.size(0), device=device),
        num_nodes=P
    )

    edge_index_sub, edge_weight_sub = remove_self_loops(edge_index_sub, edge_weight_sub)

    return edge_index_sub, edge_weight_sub


def build_supernode_graph_sampled_safe(data):
    """Build a supernode graph that tolerates partial node coverage.

    Args:
        data: ``SubgraphData`` object whose sampled subgraphs may leave original
            nodes uncovered.

    Returns:
        A tuple ``(edge_index_sub, edge_weight_sub)`` describing supernode
        connectivity induced by original graph edges across sampled subgraphs.
    """
    P = int(data.num_subgraphs)
    device = data.sg_batch.device

    if P == 0:
        return (
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, device=device),
        )

    # Build memberships: original node -> list of supernodes containing it.
    memberships = {}
    for n, s in zip(data.sg_nodes_orig.tolist(), data.sg_batch.tolist()):
        memberships.setdefault(int(n), set()).add(int(s))

    edge_counts = {}
    src_all = data.edge_index[0].tolist()
    dst_all = data.edge_index[1].tolist()

    # Expensive but robust: keep all connectivity implied by overlap memberships.
    for u, v in zip(src_all, dst_all):
        su_set = memberships.get(int(u))
        sv_set = memberships.get(int(v))
        if not su_set or not sv_set:
            continue
        for su in su_set:
            for sv in sv_set:
                if su == sv:
                    continue
                edge_counts[(su, sv)] = edge_counts.get((su, sv), 0.0) + 1.0
                edge_counts[(sv, su)] = edge_counts.get((sv, su), 0.0) + 1.0

    if not edge_counts:
        return (
            torch.empty(2, 0, dtype=torch.long, device=device),
            torch.empty(0, device=device),
        )

    row = torch.tensor([k[0] for k in edge_counts.keys()], dtype=torch.long, device=device)
    col = torch.tensor([k[1] for k in edge_counts.keys()], dtype=torch.long, device=device)
    weight = torch.tensor(list(edge_counts.values()), dtype=torch.float, device=device)

    from torch_geometric.utils import coalesce, remove_self_loops

    edge_index_sub, edge_weight_sub = coalesce(
        torch.stack([row, col], dim=0),
        weight,
        num_nodes=P,
    )
    edge_index_sub, edge_weight_sub = remove_self_loops(edge_index_sub, edge_weight_sub)
    return edge_index_sub, edge_weight_sub
