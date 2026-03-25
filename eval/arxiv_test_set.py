"""Static test set for arxiv app evaluation.

Each paper has ground truth metadata for measuring search quality,
content extraction quality, and background worker relevance.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PaperSpec:
    arxiv_id: str
    title: str
    expected_categories: list[str]
    ground_truth_keywords: list[str]
    human_summary: str
    paper_type: str  # "landmark" | "recent" | "math_heavy" | "survey"
    search_queries: list[str]
    expected_sections: list[str]
    min_char_count: int
    qa_questions: list[str] = field(default_factory=list)  # For answer_queries eval


@dataclass
class SearchQuery:
    query: str
    expected_paper_ids: list[str]
    categories: list[str] | None
    description: str


# ---------------------------------------------------------------------------
# Test papers (10)
# ---------------------------------------------------------------------------

TEST_PAPERS: list[PaperSpec] = [
    # --- Landmark papers ---
    PaperSpec(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        expected_categories=["cs.CL", "cs.LG"],
        ground_truth_keywords=[
            "transformer",
            "attention",
            "self-attention",
            "encoder",
            "decoder",
            "multi-head",
            "positional encoding",
            "BLEU",
        ],
        human_summary=(
            "Introduces the Transformer architecture based entirely on attention mechanisms, "
            "dispensing with recurrence and convolutions. Achieves state-of-the-art results on "
            "English-to-German and English-to-French translation benchmarks."
        ),
        paper_type="landmark",
        search_queries=[
            "attention is all you need transformer",
            "transformer attention mechanism",
        ],
        expected_sections=["Introduction", "Model Architecture", "Training", "Results", "Conclusion"],
        min_char_count=30000,
        qa_questions=["How many attention heads are used in the base model?", "What is the main advantage over recurrent models?"],
    ),
    PaperSpec(
        arxiv_id="1810.04805",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        expected_categories=["cs.CL"],
        ground_truth_keywords=[
            "BERT",
            "masked language model",
            "pre-training",
            "fine-tuning",
            "bidirectional",
            "next sentence prediction",
            "transformer",
        ],
        human_summary=(
            "Proposes BERT, a language representation model that pre-trains deep bidirectional "
            "transformers using masked language modeling and next sentence prediction. Sets new "
            "state-of-the-art on eleven NLP tasks."
        ),
        paper_type="landmark",
        search_queries=[
            "BERT pre-training bidirectional transformers",
            "masked language model pre-training",
        ],
        expected_sections=["Introduction", "Related Work", "BERT", "Experiments", "Conclusion"],
        min_char_count=25000,
        qa_questions=["What are the two pre-training objectives?", "How many parameters does BERT-Large have?"],
    ),
    PaperSpec(
        arxiv_id="2005.14165",
        title="Language Models are Few-Shot Learners",
        expected_categories=["cs.CL"],
        ground_truth_keywords=[
            "GPT-3",
            "few-shot",
            "language model",
            "in-context learning",
            "scaling",
            "zero-shot",
            "autoregressive",
        ],
        human_summary=(
            "Demonstrates that scaling language models to 175 billion parameters (GPT-3) "
            "enables strong few-shot performance on many NLP tasks without gradient updates, "
            "using only natural language prompts and a few demonstrations."
        ),
        paper_type="landmark",
        search_queries=[
            "GPT-3 few-shot learners",
            "language models few-shot learning scaling",
        ],
        expected_sections=["Introduction", "Approach", "Results", "Broader Impacts"],
        min_char_count=50000,
        qa_questions=["How many parameters does the largest GPT-3 model have?", "What is in-context learning?"],
    ),
    PaperSpec(
        arxiv_id="2010.11929",
        title="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        expected_categories=["cs.CV", "cs.AI", "cs.LG"],
        ground_truth_keywords=[
            "vision transformer",
            "ViT",
            "patch",
            "image classification",
            "attention",
            "ImageNet",
            "pre-training",
        ],
        human_summary=(
            "Applies a pure Transformer architecture directly to sequences of image patches "
            "for image classification. When pre-trained on large datasets, Vision Transformer "
            "(ViT) matches or exceeds state-of-the-art CNNs."
        ),
        paper_type="landmark",
        search_queries=[
            "vision transformer image recognition",
            "ViT image classification transformer",
        ],
        expected_sections=["Introduction", "Related Work", "Method", "Experiments", "Conclusion"],
        min_char_count=25000,
        qa_questions=["What patch size is used for tokenizing images?", "What dataset is used for pre-training?"],
    ),
    # --- Recent papers (late 2025) ---
    PaperSpec(
        arxiv_id="2512.02556",
        title="DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models",
        expected_categories=["cs.CL", "cs.AI"],
        ground_truth_keywords=[
            "DeepSeek",
            "language model",
            "reinforcement learning",
            "reasoning",
            "open-source",
            "mixture of experts",
            "scaling",
        ],
        human_summary=(
            "Presents DeepSeek-V3.2, an open-source large language model that harmonizes "
            "computational efficiency with superior reasoning and agent performance through "
            "a scalable reinforcement learning framework."
        ),
        paper_type="recent",
        search_queries=[
            "DeepSeek open language model",
            "open source language model scaling",
        ],
        expected_sections=["Introduction", "Architecture", "Training", "Evaluation", "Conclusion"],
        min_char_count=20000,
        qa_questions=["What training framework is used?", "What is the model architecture based on?"],
    ),
    PaperSpec(
        arxiv_id="2511.08522",
        title="AlphaResearch: Accelerating New Algorithm Discovery with Language Models",
        expected_categories=["cs.AI", "cs.CL"],
        ground_truth_keywords=[
            "algorithm discovery",
            "language model",
            "agent",
            "verification",
            "peer review",
            "autonomous",
            "research",
        ],
        human_summary=(
            "Introduces AlphaResearch, an autonomous research agent that discovers new "
            "algorithms by combining execution-based verification with a simulated peer "
            "review environment, iteratively proposing and optimizing solutions."
        ),
        paper_type="recent",
        search_queries=[
            "algorithm discovery language model",
            "autonomous research agent",
        ],
        expected_sections=["Introduction", "Method", "Experiments", "Results", "Conclusion"],
        min_char_count=15000,
        qa_questions=["How does the agent verify its proposed algorithms?", "What is the role of the peer review environment?"],
    ),
    # --- Math/figure-heavy papers ---
    PaperSpec(
        arxiv_id="1512.03385",
        title="Deep Residual Learning for Image Recognition",
        expected_categories=["cs.CV"],
        ground_truth_keywords=[
            "residual",
            "skip connection",
            "ImageNet",
            "batch normalization",
            "deep learning",
            "degradation",
            "shortcut",
        ],
        human_summary=(
            "Introduces residual learning with skip connections, enabling training of "
            "networks with over 100 layers. Wins ImageNet 2015 with 3.57% top-5 error "
            "rate, showing that deeper networks can be easier to optimize."
        ),
        paper_type="math_heavy",
        search_queries=[
            "deep residual learning image recognition",
            "ResNet skip connection",
        ],
        expected_sections=["Introduction", "Related Work", "Deep Residual Learning", "Experiments"],
        min_char_count=20000,
        qa_questions=["What problem does residual learning address?", "What is the depth of the deepest network tested?"],
    ),
    PaperSpec(
        arxiv_id="2207.00747",
        title="Elucidating the Design Space of Diffusion-Based Generative Models",
        expected_categories=["cs.CV", "cs.AI", "cs.LG", "stat.ML"],
        ground_truth_keywords=[
            "diffusion",
            "score",
            "denoising",
            "sampling",
            "noise schedule",
            "ODE",
            "stochastic",
            "generative",
        ],
        human_summary=(
            "Systematically analyzes the design space of diffusion-based generative models, "
            "identifying key design choices for the noise schedule, network architecture, and "
            "sampling procedure, achieving record FID scores on CIFAR-10 and ImageNet-64."
        ),
        paper_type="math_heavy",
        search_queries=[
            "diffusion generative model design space",
            "elucidating diffusion design",
        ],
        expected_sections=["Introduction", "Background", "Practical Improvements", "Experiments"],
        min_char_count=20000,
        qa_questions=["What FID score is achieved on CIFAR-10?", "What are the key design choices analyzed?"],
    ),
    # --- Survey papers ---
    PaperSpec(
        arxiv_id="2303.18223",
        title="A Survey of Large Language Models",
        expected_categories=["cs.CL", "cs.AI"],
        ground_truth_keywords=[
            "survey",
            "pre-training",
            "alignment",
            "RLHF",
            "instruction tuning",
            "emergent abilities",
            "scaling law",
            "GPT",
        ],
        human_summary=(
            "Comprehensive survey of large language models covering pre-training, "
            "adaptation tuning, utilization, and capacity evaluation. Discusses emergent "
            "abilities, alignment techniques, and practical deployment considerations."
        ),
        paper_type="survey",
        search_queries=[
            "survey large language models",
            "large language model survey comprehensive",
        ],
        expected_sections=["Introduction", "Overview", "Pre-Training", "Conclusion"],
        min_char_count=80000,
        qa_questions=["What alignment techniques are discussed?", "What are emergent abilities in LLMs?"],
    ),
    PaperSpec(
        arxiv_id="2308.10620",
        title="A Survey on Large Language Model based Autonomous Agents",
        expected_categories=["cs.AI", "cs.CL"],
        ground_truth_keywords=[
            "agent",
            "planning",
            "memory",
            "tool use",
            "multi-agent",
            "reasoning",
            "LLM",
            "autonomous",
        ],
        human_summary=(
            "Surveys LLM-based autonomous agents covering agent architecture (profiling, "
            "memory, planning, action), applications in social science, natural science, "
            "and engineering, and evaluation strategies."
        ),
        paper_type="survey",
        search_queries=[
            "LLM autonomous agents survey",
            "large language model agent planning memory",
        ],
        expected_sections=["Introduction", "Agent Architecture", "Application", "Evaluation"],
        min_char_count=50000,
        qa_questions=["What are the main components of an LLM-based agent?", "What application domains are covered?"],
    ),
]


# ---------------------------------------------------------------------------
# Search queries with expected results
# ---------------------------------------------------------------------------

SEARCH_QUERIES: list[SearchQuery] = [
    SearchQuery(
        query="transformer attention mechanism",
        expected_paper_ids=["1706.03762", "1810.04805"],
        categories=["cs.CL"],
        description="Should find the foundational transformer and BERT papers",
    ),
    SearchQuery(
        query="image classification deep learning residual",
        expected_paper_ids=["1512.03385", "2010.11929"],
        categories=["cs.CV"],
        description="Should find ResNet and ViT papers",
    ),
    SearchQuery(
        query="large language model survey",
        expected_paper_ids=["2303.18223"],
        categories=None,
        description="Should find the LLM survey paper",
    ),
    SearchQuery(
        query="few-shot learning language model scaling",
        expected_paper_ids=["2005.14165"],
        categories=["cs.CL"],
        description="Should find the GPT-3 paper",
    ),
    SearchQuery(
        query="diffusion generative model denoising",
        expected_paper_ids=["2207.00747"],
        categories=None,
        description="Should find the Karras diffusion paper",
    ),
    SearchQuery(
        query="open source language model reinforcement learning",
        expected_paper_ids=["2512.02556"],
        categories=None,
        description="Should find DeepSeek-V3.2 (tests recency)",
    ),
    SearchQuery(
        query="autonomous agent algorithm discovery",
        expected_paper_ids=["2511.08522"],
        categories=None,
        description="Should find AlphaResearch (tests recency)",
    ),
    SearchQuery(
        query="LLM agent planning memory tool",
        expected_paper_ids=["2308.10620"],
        categories=["cs.AI"],
        description="Should find the autonomous agents survey",
    ),
]


# ---------------------------------------------------------------------------
# Research interests for background worker eval
# ---------------------------------------------------------------------------

BG_WORKER_INTERESTS: list[str] = [
    "transformer architectures",
    "diffusion models",
    "large language model agents",
]
