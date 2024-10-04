import equinox as eqx
import jax
import jax.numpy as jnp


class MPEdgeNodeBlock(eqx.Module):
    """this model only works on undirected graph"""

    projection_node: eqx.nn.Linear
    projection_edge: eqx.nn.Linear
    edge_mlp: list
    node_mlp: list

    def __init__(
        self,
        in_node_feat,
        out_node_feat,
        in_edge_feat,
        out_edge_feat,
        num_mlp,
        key,
    ):
        keys = jax.random.split(key, 3)
        self.projection_node = eqx.nn.Linear(
            in_node_feat, out_node_feat, key=keys[0]
        )
        self.projection_edge = eqx.nn.Linear(
            in_edge_feat, out_edge_feat, key=keys[1]
        )
        self.edge_mlp = []
        self.node_mlp = []

        key = keys[2]
        h_node_feat = out_node_feat * 2 + out_edge_feat
        h_edge_feat = out_edge_feat + 2 * out_node_feat

        print(out_node_feat, out_edge_feat, "out_node_feat, out_edge_feat")
        for i in range(num_mlp):
            _, key = jax.random.split(key, 2)
            self.edge_mlp.append(
                eqx.nn.Linear(h_edge_feat, h_edge_feat, key=key)
            )
            self.edge_mlp.append(eqx.nn.PReLU())
            self.node_mlp.append(
                eqx.nn.Linear(h_node_feat, h_node_feat, key=key)
            )
            self.node_mlp.append(eqx.nn.PReLU())

        self.edge_mlp.append(
            eqx.nn.Linear(h_edge_feat, out_edge_feat, key=key)
        )
        self.node_mlp.append(
            eqx.nn.Linear(h_node_feat, out_node_feat, key=key)
        )

    def __call__(self, node_feats, edge_feats, adj_matrix):
        """adj matrix uses sparse representation here
        which has shape (E, 2) where E is the number of edges

        edge_feats: (E, feat_edge_dim)
        node_feats: (N, feat_node_dim)
        """

        # separate real and imaginary parts
        node_feats_real = node_feats.real
        node_feats_imag = node_feats.imag
        edge_feats_real = edge_feats.real
        edge_feats_imag = edge_feats.imag

        # project node and edge features to high dims
        node_feats_real = jax.vmap(self.projection_node)(node_feats_real)
        node_feats_imag = jax.vmap(self.projection_node)(node_feats_imag)
        edge_feats_real = jax.vmap(self.projection_edge)(edge_feats_real)
        edge_feats_imag = jax.vmap(self.projection_edge)(edge_feats_imag)

        prj_node_feats = node_feats_real + 1j * node_feats_imag
        prj_edge_feats = edge_feats_real + 1j * edge_feats_imag

        # aggregate node and edge features for message passing
        agg_node_feats = self.aggregate_node_feat(
            prj_node_feats, prj_edge_feats, adj_matrix
        )
        agg_node_feats_real = agg_node_feats.real
        agg_node_feats_imag = agg_node_feats.imag

        # update node features
        for layer in self.node_mlp:
            agg_node_feats_real = jax.vmap(layer)(agg_node_feats_real)
            agg_node_feats_imag = jax.vmap(layer)(agg_node_feats_imag)
        agg_node_feats = agg_node_feats_real + 1j * agg_node_feats_imag

        # aggregate edge features and updated node features
        agg_edge_feats = self.aggregate_edge_feat(
            agg_node_feats, prj_edge_feats, adj_matrix
        )
        agg_edge_feats_real = agg_edge_feats.real
        agg_edge_feats_imag = agg_edge_feats.imag

        # update edge features
        for layer in self.edge_mlp:
            agg_edge_feats_real = jax.vmap(layer)(agg_edge_feats_real)
            agg_edge_feats_imag = jax.vmap(layer)(agg_edge_feats_imag)
        agg_edge_feats = agg_edge_feats_real + 1j * agg_edge_feats_imag
        return agg_node_feats, agg_edge_feats

    def aggregate_node_feat(self, node_feats, edge_feats, adj_matrix):
        node_feat_sum = adj_matrix @ node_feats
        indices = adj_matrix.indices
        row_idx = indices[:, 0]
        edge_feat_sum = jax.ops.segment_sum(edge_feats, row_idx)
        aggreated_feat = jnp.concatenate(
            [node_feats, node_feat_sum, edge_feat_sum], axis=1
        )
        return aggreated_feat

    def aggregate_edge_feat(self, node_feats, edge_feats, adj_matrix):
        indices = adj_matrix.indices
        row_idx = indices[:, 0]
        col_idx = indices[:, 1]
        v_i = node_feats[row_idx]  # shape (E, feat_node_dim)
        v_j = node_feats[col_idx]
        aggregated_edge_feat = jnp.concatenate([edge_feats, v_i, v_j], axis=1)
        return aggregated_edge_feat


class MPNodeEdgeModel(eqx.Module):
    MPBlock: list
    alpha: jnp.ndarray 

    def __init__(self, num_mp_blocks, node_dim, edge_dim, hn_dim, he_dim, key):
        self.MPBlock = []
        self.alpha = jnp.array([0.0])
        for i in range(num_mp_blocks):
            _, key = jax.random.split(key, 2)
            if i == 0:
                self.MPBlock.append(
                    MPEdgeNodeBlock(
                        in_node_feat=node_dim,
                        out_node_feat=hn_dim,
                        in_edge_feat=edge_dim,
                        out_edge_feat=he_dim,
                        num_mlp=3,
                        key=key,
                    )
                )
            elif i < num_mp_blocks - 1:
                self.MPBlock.append(
                    MPEdgeNodeBlock(
                        in_node_feat=hn_dim,
                        out_node_feat=hn_dim,
                        in_edge_feat=he_dim,
                        out_edge_feat=he_dim,
                        num_mlp=3,
                        key=key,
                    )
                )
            else:
                self.MPBlock.append(
                    MPEdgeNodeBlock(
                        in_node_feat=hn_dim,
                        out_node_feat=1,
                        in_edge_feat=he_dim,
                        out_edge_feat=1,
                        num_mlp=3,
                        key=key,
                    )
                )

    def __call__(self, node_feats, edge_feats, adj_matrix):
        for block in self.MPBlock:
            node_feats, edge_feats = block(node_feats, edge_feats, adj_matrix)
        return node_feats, edge_feats


class GATLayer(eqx.Module):
    projection: eqx.nn.Linear
    a: jnp.ndarray
    num_heads: int

    def __init__(self, num_heads, in_feat, out_feat, key):
        key = jax.random.split(key, 2)
        out_feat = out_feat // num_heads
        self.projection = eqx.nn.Linear(
            in_feat, out_feat * num_heads, key=key[0]
        )
        self.a = jax.random.normal(key[1], (num_heads, 2 * out_feat))
        self.num_heads = num_heads

    def __call__(self, x, adj_matrix):
        num_nodes = x.shape[0]
        node_feats = jax.vmap(self.projection)(x)  # element-wise
        node_feats = node_feats.reshape(num_nodes, self.num_heads, -1)

        logit_parent = (
            node_feats * self.a[None, :, : self.a.shape[1] // 2]
        ).sum(axis=-1)
        logit_child = (
            node_feats * self.a[None, :, self.a.shape[1] // 2 :]
        ).sum(axis=-1)
        attn_logits = (
            logit_parent[:, None, :] + logit_child[None, :, :]
        )  # broadcasting to all node pairs
        attn_logits = jax.nn.leaky_relu(attn_logits, 0.2)

        # Mask out nodes that do not have an edge between them
        attn_logits = jnp.where(
            jnp.abs(adj_matrix[..., None]) != 0.0,
            attn_logits,
            jnp.ones_like(attn_logits) * 0.0,
        )
        attn_probs = jax.nn.softmax(attn_logits, axis=1)
        node_feats = jnp.einsum("ijh,jhc->ihc", attn_probs, node_feats)
        node_feats = node_feats.reshape(num_nodes, -1)
        return node_feats


class GAT(eqx.Module):
    layers: list

    def __init__(self, num_heads, in_feat, out_feat, num_layers, key):
        key = jax.random.split(key, num_layers)
        self.layers = [GATLayer(num_heads, in_feat, out_feat, key[0])]
        for i in range(1, num_layers):
            self.layers.append(GATLayer(num_heads, out_feat, out_feat, key[i]))

    def __call__(self, x, adj_matrix):
        x_real, x_imag = x.real, x.imag
        for layer in self.layers:
            x_real = layer(x_real, adj_matrix)
            x_imag = layer(x_imag, adj_matrix)
        x = x_real + 1j * x_imag
        return x


class GraphPrecond(eqx.Module):
    graphnet: eqx.Module
    dense_layers: list
    alpha: jnp.ndarray
    n_multiples: int

    def __init__(
        self,
        num_heads,
        in_feat,
        out_feat,
        num_layers,
        num_dense_layers,
        dense_h_dim,
        n_multiples,
        key,
    ):

        self.n_multiples = (
            n_multiples  # number of sets of gamma matrix equivalent
        )

        self.graphnet = GAT(num_heads, in_feat, out_feat, num_layers, key)
        self.dense_layers = []
        self.alpha = jnp.array([0.0])

        for i in range(num_dense_layers):
            _, key = jax.random.split(key, 2)
            if i == 0:
                self.dense_layers.append(
                    eqx.nn.Linear(out_feat * 128, dense_h_dim, key=key)
                )
            else:
                self.dense_layers.append(
                    eqx.nn.Linear(dense_h_dim, dense_h_dim, key=key)
                )
            self.dense_layers.append(eqx.nn.PReLU())

        self.dense_layers.append(
            eqx.nn.Linear(dense_h_dim, 2**3 * self.n_multiples, key=key)
        )

    def __call__(self, x, adj_matrix):
        print(x.shape, adj_matrix.shape, "inspecting shapes")
        x = self.graphnet(x, adj_matrix)
        # pooling
        x = x.ravel()

        x_real, x_imag = x.real, x.imag
        for layer in self.dense_layers:
            x_real = layer(x_real)
            x_imag = layer(x_imag)
        x = x_real + 1j * x_imag
        x = x.reshape(2 * self.n_multiples, 2, 2)
        return x


class GATPrecond(eqx.Module):
    """
    The network takes U1 as input and outputs
    the non-zero entries of the preconditioner matirx
    which has the same structure as the original matrix
    """

    graphnet: eqx.Module
    dense_layers: list
    alpha: jnp.ndarray

    def __init__(
        self,
        num_heads,
        in_feat,
        out_feat,
        num_layers,
        num_dense_layers,
        dense_h_dim,
        num_nnzs,
        key,
    ):

        self.graphnet = GAT(num_heads, in_feat, out_feat, num_layers, key)
        self.dense_layers = []
        self.alpha = jnp.array([0.0])

        for i in range(num_dense_layers):
            _, key = jax.random.split(key, 2)
            if i == 0:
                self.dense_layers.append(
                    eqx.nn.Linear(out_feat * 128, dense_h_dim, key=key)
                )
            else:
                self.dense_layers.append(
                    eqx.nn.Linear(dense_h_dim, dense_h_dim, key=key)
                )
            self.dense_layers.append(eqx.nn.PReLU())

        self.dense_layers.append(eqx.nn.Linear(dense_h_dim, num_nnzs, key=key))

    def __call__(self, x, adj_matrix):
        x = self.graphnet(x, adj_matrix)
        # pooling
        x = x.ravel()

        x_real, x_imag = x.real, x.imag
        for layer in self.dense_layers:
            x_real = layer(x_real)
            x_imag = layer(x_imag)
        x = x_real + 1j * x_imag
        return x


if __name__ == "__main__":
    from jax.experimental import sparse

    # test
    key = jax.random.PRNGKey(0)

    node_feats = jax.random.normal(key, (32, 128, 1)).astype(jnp.complex64)
    adj_matrix = jax.random.normal(key, (128, 128)).astype(jnp.complex64)
    adj_matrix = sparse.BCOO.fromdense(adj_matrix)
    edge_feats = jax.random.normal(
        key, (32, len(adj_matrix.indices), 1)
    ).astype(jnp.complex64)

    print(node_feats.shape, adj_matrix.shape, edge_feats.shape)

    # graph_precond = MPEdgeNodeBlock(
    #     in_node_feat=1,
    #     out_node_feat=16,
    #     in_edge_feat=1,
    #     out_edge_feat=16,
    #     num_mlp=3,
    #     key=key,
    # )

    graph_precond = MPNodeEdgeModel(3, 1, 1, 8, 8, key)

    node, edge = eqx.filter_jit(jax.vmap(graph_precond, in_axes=(0, 0, None)))(
        node_feats, edge_feats, adj_matrix
    )
    print(node.shape, edge.shape, "node, edge")
