import tensorflow as tf
import tensorflow_models as tfm

class TFMCrossAttentionPooling(tf.keras.layers.Layer):
    """Cross-attention pooling with learned query tokens"""
    def __init__(self,
                 num_heads: int,
                 key_dim: int,
                 num_query_tokens: int = 4,
                 value_dim: int = None,
                 dropout: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim or key_dim
        self.num_query_tokens = num_query_tokens
        self.dropout = dropout
        
        # Multi-head attention layer
        self.cross_attention = tfm.nlp.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=self.value_dim,
            dropout=dropout,
            use_bias=True
        )
        
        # Layer norm and MLP for processing
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(key_dim * 2, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(key_dim)
        ])
        
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        
        # Learnable query tokens for pooling
        self.query_tokens = self.add_weight(
            name='query_tokens',
            shape=(1, self.num_query_tokens, feature_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, x, training=False, attention_mask=None):
        batch_size = tf.shape(x)[0]
        
        # Expand query tokens for batch
        queries = tf.tile(self.query_tokens, [batch_size, 1, 1])
        
        # Cross attention: queries attend to input sequence
        attended_output, attention_weights = self.cross_attention(
            query=queries,
            value=x,
            key=x,
            attention_mask=attention_mask,
            training=training,
            return_attention_scores=True
        )
        
        # Apply layer norm and MLP
        attended_output = self.layer_norm(attended_output)
        processed_output = self.mlp(attended_output, training=training)
        
        # Pool the query tokens (you can use mean, max, or learned pooling)
        pooled = tf.reduce_mean(processed_output, axis=1)  # (batch_size, feature_dim)
        
        return pooled, attention_weights
