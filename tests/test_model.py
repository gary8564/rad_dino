import unittest
import torch
from unittest.mock import patch, MagicMock
from rad_dino.models.model import DinoClassifier

class MockBackboneOutput:
    def __init__(self, hidden_state, attentions=None):
        self.last_hidden_state = hidden_state
        if attentions is None:
            # Create mock attentions tensor with proper shape
            # For DINOv2-base: 12 layers, 12 heads, sequence length based on image patches
            batch_size = hidden_state.shape[0]
            num_layers = 12
            num_heads = 12
            seq_len = hidden_state.shape[1]  # Should be 197 for 224x224 image (196 patches + 1 CLS)
            attentions = [torch.randn(batch_size, num_heads, seq_len, seq_len) for _ in range(num_layers)]
        self.attentions = attentions

class TestDinoClassifier(unittest.TestCase):
    def setUp(self):
        # Create a mock backbone
        self.mock_backbone = MagicMock()
        self.mock_backbone.config.hidden_size = 768  # DINOv2 base hidden size
        self.num_classes = 2
        self.model = DinoClassifier(self.mock_backbone, num_classes=self.num_classes)
        self.batch_size = 4
        self.input_size = 224  # Standard input size for DINOv2
        
    def test_forward_pass(self):
        """Test forward pass through the model."""
        x = torch.randn(self.batch_size, 3, self.input_size, self.input_size)
        
        # Mock backbone to return an object with last_hidden_state and attentions
        # Shape: (batch_size, sequence_length, hidden_size)
        expected_backbone_output = torch.randn(self.batch_size, 197, 768) 
        self.mock_backbone.return_value = MockBackboneOutput(expected_backbone_output)
        
        logits, attentions = self.model(x)
        
        # Check output shapes
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attentions.shape, (12, self.batch_size, 12, 197, 197))  # num_layers, batch, heads, seq_len, seq_len
        
        # Verify backbone was called with correct arguments
        self.mock_backbone.assert_called_once_with(
            x, 
            output_attentions=True, 
            return_dict=True, 
            interpolate_pos_encoding=True
        )
        
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        # Test with standard size
        x_standard = torch.randn(self.batch_size, 3, self.input_size, self.input_size)
        expected_output_standard = torch.randn(self.batch_size, 197, 768)
        self.mock_backbone.return_value = MockBackboneOutput(expected_output_standard)
        logits_standard, attentions_standard = self.model(x_standard)
        self.assertEqual(logits_standard.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attentions_standard.shape, (12, self.batch_size, 12, 197, 197))
        
        # Test with different size
        x_different = torch.randn(self.batch_size, 3, 256, 256)
        expected_output_different = torch.randn(self.batch_size, 197, 768)
        self.mock_backbone.return_value = MockBackboneOutput(expected_output_different)
        logits_different, attentions_different = self.model(x_different)
        self.assertEqual(logits_different.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attentions_different.shape, (12, self.batch_size, 12, 197, 197))
        
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        # Create input tensor with requires_grad=True
        x = torch.randn(self.batch_size, 3, self.input_size, self.input_size, requires_grad=True)
        
        # Create a function that will be used to mock the backbone
        def mock_backbone_forward(input_tensor, output_attentions=True, return_dict=True, interpolate_pos_encoding=True):
            # Create output that depends on input to maintain gradient flow
            hidden = input_tensor.mean(dim=[2, 3], keepdim=True)  # [B, 3, 1, 1]
            hidden = hidden.expand(-1, -1, 197, 768)  # [B, 3, 197, 768]
            hidden = hidden.mean(dim=1)  # [B, 197, 768]
            return MockBackboneOutput(hidden)
        
        # Set up the mock to use our function
        self.mock_backbone.side_effect = mock_backbone_forward
        
        # Forward pass
        logits, attentions = self.model(x)
        
        # Backward pass with a scalar loss
        loss = logits.mean()  # Use mean instead of sum for better numerical stability
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(x.grad, "Input gradients should not be None")
        self.assertIsNotNone(self.model.head[0].weight.grad, "Head weight gradients should not be None")
        self.assertIsNotNone(self.model.head[0].bias.grad, "Head bias gradients should not be None")
        
        # Verify gradient shapes
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(self.model.head[0].weight.grad.shape, self.model.head[0].weight.shape)
        self.assertEqual(self.model.head[0].bias.grad.shape, self.model.head[0].bias.shape)
        
    def test_model_parameters(self):
        """Test model parameter initialization and count."""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        head_params = sum(p.numel() for p in self.model.head.parameters())
        
        # Head should have (768 * num_classes + num_classes) parameters
        expected_head_params = 768 * self.num_classes + self.num_classes
        self.assertEqual(head_params, expected_head_params)
        
        # Total parameters should be at least head parameters
        self.assertGreaterEqual(total_params, head_params)

if __name__ == "__main__":
    unittest.main()
