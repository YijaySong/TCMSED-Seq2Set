"""Legacy TCN and PatchTST-style comparisons on the unified CV splits."""

from run_modern_encoders import PatchTSTStyleEncoder, SequenceSetModel, TCNEncoder


def build(method, vocab, k):
    """Use the original encoder and ADD/STOP decoder implementations."""
    if method == "tcn_sequence":
        encoder = TCNEncoder(vocab)
    elif method == "patchtst_style_sequence":
        encoder = PatchTSTStyleEncoder(vocab)
    else:
        raise ValueError(f"Unknown modern encoder method: {method}")
    return SequenceSetModel(encoder, k)
