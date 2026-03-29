import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuroFuzzySIEM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, num_numeric_features=5, latent_dim=16, num_rules=32):
        super(NeuroFuzzySIEM, self).__init__()
        self.latent_dim = latent_dim
        self.num_rules = num_rules
        
        # 1. Neural Representation Layer
        # Sequence of categorical inputs -> embedded and aggregated
        self.embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embedding_dim, padding_idx=0)
        
        # Linear layers for neural mapping
        # Output of embedding is 10 * embedding_dim (because window_size=10)
        self.fc1 = nn.Linear(10 * embedding_dim + num_numeric_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, latent_dim)
        
        # 2. Fuzzification Layer
        # 3 Gaussian sets per latent dimension: Low, Medium, High
        self.num_fuzzy_sets = 3
        # Learnable means and stdevs
        self.mu = nn.Parameter(torch.randn(latent_dim, self.num_fuzzy_sets))
        self.sigma = nn.Parameter(torch.ones(latent_dim, self.num_fuzzy_sets))
        
        # 3. Fuzzy Rule Layer
        # Compute K rules from latent fuzzy sets
        # A simple linear combination from the 16*3 fuzzy outputs to num_rules
        self.rule_weights = nn.Parameter(torch.randn(latent_dim * self.num_fuzzy_sets, num_rules))
        
        # 4. Defuzzification Layer
        # Consequents for each rule
        self.rule_consequents = nn.Parameter(torch.rand(num_rules))
        
    def fuzzify(self, x):
        # x is (batch, latent_dim)
        # Output is (batch, latent_dim, num_fuzzy_sets)
        # exp(-0.5 * ((x - mu)/sigma)^2)
        x_expanded = x.unsqueeze(2) # (batch, latent_dim, 1)
        mu_expanded = self.mu.unsqueeze(0) # (1, latent_dim, num_fuzzy_sets)
        sigma_expanded = self.sigma.unsqueeze(0) + 1e-6 # avoid division by zero
        
        memberships = torch.exp(-0.5 * ((x_expanded - mu_expanded) / sigma_expanded) ** 2)
        return memberships
        
    def forward(self, x_cat, x_num):
        # x_cat: (batch, window_size)
        # x_num: (batch, num_numeric_features)
        
        batch_size = x_cat.size(0)
        
        # Embedded categorical sequence
        embeds = self.embedding(x_cat) # (batch, 10, embedding_dim)
        embeds_flat = embeds.view(batch_size, -1) # (batch, 10 * embedding_dim)
        
        # Concatenate with numerical features
        x = torch.cat([embeds_flat, x_num], dim=1)
        
        # Neural Representation Layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        latent = self.fc3(x) # (batch, 16)
        
        # Fuzzification Layer
        memberships = self.fuzzify(latent) # (batch, 16, 3)
        memberships_flat = memberships.view(batch_size, -1) # (batch, 48)
        
        # Fuzzy Rule Layer (Soft combination of rule antecedents)
        # softmax over rules to normalize rule strengths
        rule_firings = torch.matmul(memberships_flat, self.rule_weights) # (batch, num_rules)
        rule_firings = F.softmax(rule_firings, dim=1)
        
        # Defuzzification Layer
        # Weighted average of rule consequences
        risk_score = torch.sum(rule_firings * self.rule_consequents, dim=1) # (batch)
        
        # Scale to 0-100 continuously
        risk_score = torch.sigmoid(risk_score) * 100.0
        
        return risk_score

if __name__ == '__main__':
    # Test model pass
    model = NeuroFuzzySIEM(vocab_size=30)
    x_cat = torch.randint(0, 30, (8, 10))
    x_num = torch.randn(8, 5)
    out = model(x_cat, x_num)
    print("Risk score output shape:", out.shape)
    print("Scores:\n", out)
