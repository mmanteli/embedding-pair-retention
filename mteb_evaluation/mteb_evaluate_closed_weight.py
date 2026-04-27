import mteb
from mteb.cache import ResultCache
from sentence_transformers import SentenceTransformer
import sys
import os
from data_and_encoder import load_model
import numpy as np
from typing import Optional 
from mteb import EncoderProtocol, TaskMetadata
from mteb.models.model_meta import ModelMeta
from torch.utils.data import DataLoader


try:
    from mteb.encoder_interface import PromptType
except ImportError:
    try:
        from mteb import PromptType
    except ImportError:
        from typing import Any
        PromptType = Any


# get sentences from data loader to feed to openai/gemini model
# as they cannot handle data loader
def extract_sentences_from_dataloader(inputs: DataLoader, truncate=None) -> list[str]:
    all_sentences = []
    for batch in inputs:
        #print("in batch")
        if isinstance(batch, dict):
            for key in ("text", "sentence", "sentences"):
                if key in batch:
                    texts = batch[key]
                    all_sentences.extend(texts if isinstance(texts, list) else [texts])
                    break
        elif isinstance(batch, (list, tuple)):
            all_sentences.extend(batch)
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")
    if truncate:
        return truncate_sentences(all_sentences, max_chars=truncate)
    return all_sentences


def truncate_sentences(sentences: list[str], max_chars = 30000) -> list[str]:
    """
    Rough truncation for Gemini. Gemini's limit is ~2048 tokens (~30k chars worst case).
    Using this for OpenAI as well, with limit ~8191 tokens (~120k chars for ENGLISH)
    """
    return [s[:max_chars] for s in sentences]

# inherit "Encoder" from mteb
# This was a pain
class OpenAIEmbedder(EncoderProtocol):
    """
    Wrapper around OpenAI embedding API to make it compatible with MTEB.
    """

    def __init__(self, model_name: str, revision: str | None = None, device: str | None = None, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self.client = OpenAI()  # have OPENAI_API_KEY set in environment
        self._model_name = model_name.split("/", 1)[1]  # strip "openai/" prefix
        self._full_model_name = model_name  # keep for ModelMeta

    # MTEB refuses to work without model meta
    @property
    def mteb_model_meta(self):
        return ModelMeta(
            name=self._full_model_name,
            revision="1",
            release_date=None,
            languages=None,
            loader=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=False,
            public_training_code=None,
            public_training_data=None,
            framework=[],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )

    # rewrite this so get dataloader --> list of sentences for closed source models
    def encode(
        self,
        inputs: DataLoader | list[str],
        *,
        task_metadata: TaskMetadata = None,
        hf_split: str = None,
        hf_subset: str = None,
        prompt_type: Optional[PromptType] = None,
        **kwargs,
        ) -> np.ndarray:
        # FIXED: handle both DataLoader and plain list inputs
        if isinstance(inputs, DataLoader):
            sentences = extract_sentences_from_dataloader(inputs, truncate=70000)
        elif isinstance(inputs, list):
            sentences = truncate_sentences(inputs, max_chars=70000)
        else:
            raise ValueError(f"Unexpected input type: {type(inputs)}")
        # debugging
        print(f"[ENCODE] prompt_type={prompt_type}, n_sentences={len(sentences)}, "
            f"longest={max(len(s) for s in sentences)} chars")

        
        # embed
        all_embeddings = []
        batch_size = 64
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self._model_name
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return np.array(all_embeddings)


# Same for gemini
class GeminiEmbedder(EncoderProtocol):
    """
    Wrapper around Google Gemini embedding API to make it compatible with MTEB.
    """
    def __init__(self, model_name: str, revision: str | None = None, device: str | None = None, **kwargs):
        try:
            from google import genai
        except ImportError:
            raise ImportError("pip install google-genai")
        
        # FIXED: new google-genai SDK uses client pattern, not configure()
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        
        bare = model_name.split("/", 1)[1] if "/" in model_name else model_name
        self._model_name = bare  # FIXED: new SDK does NOT use "models/" prefix
        self._full_model_name = model_name

    # Same as openai
    @property
    def mteb_model_meta(self):
        return ModelMeta(
            name=self._full_model_name,
            revision="1",
            release_date=None,
            languages=None,
            loader=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=False,
            public_training_code=None,
            public_training_data=None,
            framework=[],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )

    # rewrite to get from dataloader --> list of sentences
    def encode(
        self,
        inputs: DataLoader,
        *,
        task_metadata: TaskMetadata = None,
        hf_split: str = None,
        hf_subset: str = None,
        prompt_type: Optional[PromptType] = None,
        **kwargs,
    ) -> np.ndarray:
        import time
        if isinstance(inputs, DataLoader):
            sentences = extract_sentences_from_dataloader(inputs, truncate=30000)
        elif isinstance(inputs, list):
            sentences = truncate_sentences(inputs, max_chars=30000)
        else:
            raise ValueError(f"Unexpected input type: {type(inputs)}")
        
        print(f"[ENCODE] prompt_type={prompt_type}, n_sentences={len(sentences)}, "
              f"longest={max(len(s) for s in sentences)} chars")

        # FIXED: new SDK uses client.models.embed_content()
        all_embeddings = []
        for i, s in enumerate(sentences):
            result = self.client.models.embed_content(
                model=self._model_name,
                contents=s  # FIXED: 'contents' not 'content' in new SDK
            )
            all_embeddings.append(result.embeddings[0].values)  # FIXED: new response structure
            if i > 0 and i % 50 == 0:
                time.sleep(1.0)
        return np.array(all_embeddings)

    # these need to be rewritten also due to data format
    def similarity(self, embeddings1, embeddings2):
        e1 = np.atleast_2d(embeddings1)
        e2 = np.atleast_2d(embeddings2)
        norm1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        norm2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
        return (norm1 @ norm2.T).squeeze()

    def similarity_pairwise(self, embeddings1, embeddings2):
        e1 = np.atleast_2d(embeddings1)
        e2 = np.atleast_2d(embeddings2)
        norm1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        norm2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
        return np.sum(norm1 * norm2, axis=1).squeeze()


# load closed source or local 
def load_any_model(model_name):

    if model_name.startswith("openai/"):
        print(f"[INFO] Using OpenAI API model: {model_name}")
        return OpenAIEmbedder(model_name), False

    elif model_name.startswith("gemini/"):
        print(f"[INFO] Using Gemini API model: {model_name}")
        return GeminiEmbedder(model_name), False

    else:
        print(f"[INFO] Using local SentenceTransformer model: {model_name}")
        return load_model(model_name), True



#-----------------------------------main logic------------------------------------#
assert len(sys.argv) == 2, "Give a model to evaluate"
model_name = sys.argv[1]
if os.path.exists(model_name):
    model_name = os.path.basename(model_name).replace("__", "/")

model, needs_eval = load_any_model(model_name)
if needs_eval:
    model.eval()

languages = ["fra", "zho", "ita", "deu", "jpn", "bul", "fin", "hin", "ell", "fas", "ukr", "vie", "eng"]
langauges_tatoeba = [f"{i}-eng" for i in languages if i != "eng"]


# get tatoeba filtered by language
def get_tatoeba():
    tatoeba = mteb.get_task("Tatoeba")
    subsets = None
    for attr in ("available_subsets", "hf_subsets", "subsets"):
        if hasattr(tatoeba, attr):
            subsets = getattr(tatoeba, attr)
            break
    if subsets is None:
        raise RuntimeError("Couldn't find subsets on the Tatoeba task.")
    wanted = langauges_tatoeba
    chosen = [s for s in subsets if s in wanted]
    if not chosen:
        raise RuntimeError(f"None of {wanted} found. Available: {subsets[:30]}")
    return mteb.get_task("Tatoeba", hf_subsets=chosen)

# get webfaq filtered by language
def get_webfaq(langs_to_select=None):

    webfaq = mteb.get_task("WebFAQRetrieval")
    subsets = None
    for attr in ("available_subsets", "hf_subsets", "subsets"):
        if hasattr(webfaq, attr):
            subsets = getattr(webfaq, attr)
            break
    if subsets is None:
        raise RuntimeError("Couldn't find subsets on the WebFAQRetrieval task.")
    wanted = languages if langs_to_select is None else langs_to_select
    chosen = [s for s in subsets if s in wanted]
    if not chosen:
        raise RuntimeError(f"None of {wanted} found. Available: {subsets[:30]}")
    return mteb.get_task("WebFAQRetrieval", hf_subsets=chosen)


default_tasks = mteb.get_tasks(
    tasks=["ARCChallenge", "SummEvalSummarization.v2"], 
    languages=languages
)


tatoeba_restricted = get_tatoeba()
# trying to run with one at a time
#webfaq_langs=["ell"]
#webfaq_restricted = get_webfaq(langs_to_select=webfaq_langs)
tasks = list(default_tasks)  # + [tatoeba_restricted] +   # webfaq is IMPOSSIBLE


#if tasks == [webfaq_restricted]:
#    outfile = f"mteb_out/{model_name.replace('/', '__')}__webfaq_{'-'.join(webfaq_langs)}"
#else:
outfile = f"mteb_out/{model_name.replace('/', '__')}"
os.makedirs(outfile, exist_ok=True)

cache = ResultCache(cache_path=outfile)

# CHANGED: API models pass empty encode_kwargs (they handle batching internally)
# local models still use batch_size=8 for memory management
encode_kwargs = {} if not needs_eval else {"batch_size": 8}
results = mteb.evaluate(model, tasks=tasks, cache=cache, encode_kwargs=encode_kwargs)

print(results)
print(results.task_results)