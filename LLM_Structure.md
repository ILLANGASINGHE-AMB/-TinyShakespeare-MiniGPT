                    MiniGPT
                       │
              ┌────────┴────────┐
              │                 │
       Token Embedding    Position Embedding
              │                 │
              └────────┬────────┘
                       ↓
              Transformer Block
                       ↓
              Transformer Block
                       ↓
              Transformer Block
                       ↓
              Transformer Block
                       ↓
                 LayerNorm
                       ↓
                  LM Head
                       ↓
                 63 logits