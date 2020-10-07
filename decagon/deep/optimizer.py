import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class DecagonOptimizer(object):
    def __init__(self, embeddings, latent_inters, latent_varies,
                 degrees, edge_types, edge_type2dim, placeholders,
                 margin=0.1, neg_sample_weights=1., batch_size=100):
        self.embeddings= embeddings
        self.latent_inters = latent_inters
        self.latent_varies = latent_varies
        self.edge_types = edge_types
        self.degrees = degrees
        self.edge_type2dim = edge_type2dim
        #{ngenes;ndrugs}
        self.obj_type2n = {i: self.edge_type2dim[i,j][0][0] for i, j in self.edge_types}
        self.margin = margin
        self.neg_sample_weights = neg_sample_weights
        self.batch_size = batch_size

        self.inputs = placeholders['batch']
        self.neg_inputs = placeholders['neg_batch']
        self.batch_edge_type_idx = placeholders['batch_edge_type_idx']
        self.batch_row_edge_type = placeholders['batch_row_edge_type']
        self.batch_col_edge_type = placeholders['batch_col_edge_type']

        self.row_inputs = tf.squeeze(gather_cols(self.inputs, [0]))
        self.col_inputs = tf.squeeze(gather_cols(self.inputs, [1]))
        self.neg_rows = tf.squeeze(gather_cols(self.neg_inputs, [0]))
        self.neg_cols = tf.squeeze(gather_cols(self.neg_inputs, [1]))
        # Indices for selecting the correct (drug or gene) embeddings in the predict functions
        obj_type_n = [self.obj_type2n[i] for i in range(len(self.embeddings))] #[n_genes,n_drugs]
        self.obj_type_lookup_start = tf.cumsum([0] + obj_type_n[:-1])#[0,n_genes]
        self.obj_type_lookup_end = tf.cumsum(obj_type_n)#[n_genes,n_drugs+n_genes]
        
        self.preds = self.batch_predict(self.row_inputs, self.col_inputs)
        self.outputs = tf.diag_part(self.preds)
        self.outputs = tf.reshape(self.outputs, [-1])

        self.neg_preds = self.batch_predict(self.neg_rows, self.neg_cols)
        self.neg_outputs = tf.diag_part(self.neg_preds)
        self.neg_outputs = tf.reshape(self.neg_outputs, [-1])

        self.predict()

        self._build()

    def batch_predict(self, row_inputs, col_inputs):
        ''' Returns a 2D tensor of dimension n_ixn_j with the probabilities of each pair of given
        nodes belonging to the category represented by self.batch_edge_type_idx.
        Recieves as input the row and column indices of the minibatch examples.
        '''
        concatenated = tf.concat(self.embeddings, 0)
        # Choose between drug and gene embeddings depending on the edge type
        # Rows
        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_row_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_row_edge_type)
        indices = tf.range(ind_start, ind_end)
        row_embeds = tf.gather(concatenated, indices)
        # Selects only row embeddings of nodes present in minibatch
        row_embeds = tf.gather(row_embeds, row_inputs)
        # Cols
        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_col_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_col_edge_type)
        indices = tf.range(ind_start, ind_end)
        col_embeds = tf.gather(concatenated, indices)
        # Selects only column embeddings of nodes present in minibatch
        col_embeds = tf.gather(col_embeds, col_inputs)
        # Choose the appropiate weight tensor
        latent_inter = tf.gather(self.latent_inters, self.batch_edge_type_idx)
        latent_var = tf.gather(self.latent_varies, self.batch_edge_type_idx)
        # Perform the decoder tensor multiplication
        product1 = tf.matmul(row_embeds, latent_var)
        product2 = tf.matmul(product1, latent_inter)
        product3 = tf.matmul(product2, latent_var)
        preds = tf.matmul(product3, tf.transpose(col_embeds))
        return preds

    def predict(self):
        ''' Performs the calculation of probabilities of belonging to the class given by
        self.bacth_edge_type_idx of each pair of nodes present in the whole dataset. 
        Saves the probabilities in the tensor self.predictions. It is
        the same calculation done in batch predict but in all dataset.
        '''
        concatenated = tf.concat(self.embeddings, 0)
        # Choose between drug and gene embeddings depending on the edge type
        # Rows
        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_row_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_row_edge_type)
        indices = tf.range(ind_start, ind_end)
        row_embeds = tf.gather(concatenated, indices)
        # Cols
        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_col_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_col_edge_type)
        indices = tf.range(ind_start, ind_end)
        col_embeds = tf.gather(concatenated, indices)
        # Choose the appropiate weight tensor
        latent_inter = tf.gather(self.latent_inters, self.batch_edge_type_idx)
        latent_var = tf.gather(self.latent_varies, self.batch_edge_type_idx)
        # Perform the decoder tensor multiplication
        product1 = tf.matmul(row_embeds, latent_var)
        product2 = tf.matmul(product1, latent_inter)
        product3 = tf.matmul(product2, latent_var)
        self.predictions = tf.matmul(product3, tf.transpose(col_embeds))

    def _build(self):
        self.cost = self._xent_loss(self.outputs, self.neg_outputs)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #  TO SOLVE (by SNAPS): This function makes the dense tensor
        self.opt_op = self.optimizer.minimize(self.cost)

    def _xent_loss(self, aff, neg_aff):
        """Cross-entropy optimization."""
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
        return loss


def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(
            tf.gather(p_flat, i_flat), [p_shape[0], -1])
