import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class Transformer_Decoder(nn.Module):
    """
    transformer decoder architecture for image captioning
    
    Attributes:
        embedding: embedding layer for token representations
        pos_encoding: positional encoding for sequence positions
        decoder: transformwerdecoder for sequence processing
        fc: final projection layer to vocabulary
    """
    
    def __init__(self, embed_size: int, vocab_size: int, hidden_size: int, num_layers: int):
        """
        initialize the transformer decoder
        
        Args:
            embed_size: dimension of embeddings
            vocab_size: size of vocabulary
            hidden_size: dimension of feed-forward networks
            num_layers: nr of transformer layers
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 22, embed_size))  # max_len=22
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=16,
            dim_feedforward=hidden_size,
            dropout=0.2,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features: torch.Tensor, captions: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        forward pass through the decoder
        
        Args:
            features: image features [batch_size, feature_dim]
            captions: caption token ids [batch_size, seq_len]
            mask: optional mask for decoder self-attention
            
        Returns:
            output logits [batch_size, seq_len, vocab_size]
        """
        batch_size = captions.size(0)
        seq_len = captions.size(1)

        # embed captions and add positional encoding
        embedded = self.embedding(captions) + self.pos_encoding[:, :seq_len, :]

        # causal mask for autoregressive decoding if not provided
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(features.device)

        # features as memory for cross-attention
        memory = features.unsqueeze(1)  


        output = self.decoder(tgt=embedded, memory=memory, tgt_mask=mask)
        return self.fc(output)


def sample_caption(image_feature: torch.Tensor, decoder: Transformer_Decoder, 
                  project_features: nn.Module, word2idx: dict, idx2word: dict, 
                  device: str, max_len: int = 22, temperature: float = 1.0) -> Tuple[List[int], torch.Tensor, str]:
    """
    sample a caption using the current policy (the decoder), keeping track of log probabilities
    for policy gradient training

    Args:
        image_feature: image feature tensor
        decoder: decoder model
        project_features: projection layer
        word2idx: word to index mapping
        idx2word: index to word mapping
        device: device to run on
        max_len: max caption length
        temperature: temperature for sampling (higher = more diverse)

    Returns:
        tokens: list of token ids for the sampled caption
        log_probs: sum of log probabilities for the sampled tokens
        caption: generated caption as a string
    """
    import torch.nn.functional as F
    
    decoder.eval()  # set to eval mode initially
    project_features.eval()

    # project image features
    with torch.no_grad():
        projected = project_features(image_feature.unsqueeze(0))

    decoder.train()

    # start with start of sequence token
    tokens = [word2idx["< SOS >"]]
    log_probs_list = []

    # generate tokens one by one
    for i in range(max_len - 1):  # -1 because we've already added SOS
        
        curr_seq = torch.tensor([tokens], dtype=torch.long).to(device)

        # predictions
        with torch.enable_grad():  #  tracking gradients
            outputs = decoder(projected, curr_seq)
            logits = outputs[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            dist = torch.distributions.Categorical(probs)
            next_token = dist.sample()
            log_prob = dist.log_prob(next_token)

            # add to tracking variables
            log_probs_list.append(log_prob)
            tokens.append(next_token.item())

            # stop if we encounter the EOS token
            if next_token.item() == word2idx["<EOS>"]:
                break

    # convert token IDs to words (excluding SOS, EOS, and PAD)
    caption_words = [idx2word[token] for token in tokens
                    if token not in [word2idx["<PAD>"], word2idx["< SOS >"], word2idx["<EOS>"]]]
    caption = " ".join(caption_words)


    return tokens, torch.stack(log_probs_list).sum() if log_probs_list else torch.tensor(0.0), caption