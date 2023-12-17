
from IFQRE.data.dataset import SequentialDataset, KnowledgeBasedDataset


class KGSeqDataset(SequentialDataset, KnowledgeBasedDataset):
    """Containing both processing of Sequential Models and Knowledge-based Models.

    Inherit from :class:`~IFQRE.data.dataset.sequential_dataset.SequentialDataset` and
    :class:`~IFQRE.data.dataset.kg_dataset.KnowledgeBasedDataset`.
    """

    def __init__(self, config):
        super().__init__(config)
