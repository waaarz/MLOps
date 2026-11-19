import torch
from src.model import BeanCNN

def test_model_output_shape():
    model = BeanCNN(num_classes=3, dropout=0.3)
    assert model(torch.randn(2,3,96,96)).shape == (2,3)
