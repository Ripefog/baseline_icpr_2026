try:
    from .fusion import AttentionFusion
    from .crnn import MultiFrameCRNN
except ImportError:
    from fusion import AttentionFusion
    from crnn import MultiFrameCRNN

__all__ = ['AttentionFusion', 'MultiFrameCRNN']
