# langchain-qualcomm-inference-suite

This package contains the LangChain integration with Qualcomm AI Inference Suite 

## Installation

```bash
pip install -U langchain-qualcomm-inference-suite
```

And you should configure credentials by setting the following environment variables:

1. You must set the environment variable `IMAGINE_API_KEY` to your
   personal Imagine API key.

2. You must set the environment variable `IMAGINE_ENDPOINT_URL` pointing to the
   endpoint you are using.

## Chat Models

`ChatQIS` class exposes chat models from Qualcomm Inference Suite.

```python
from langchain_qualcomm_inference_suite import ChatQIS

llm = ChatQIS()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`QISEmbeddings` class exposes embeddings from Qualcomm Inference Suite.

```python
from langchain_qualcomm_inference_suite import QISEmbeddings

embeddings = QISEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`QISLLM` class exposes LLMs from Qualcomm Inference Suite.

```python
from langchain_qualcomm_inference_suite import QISLLM

llm = QISLLM()
llm.invoke("The meaning of life is")
```

## License

langchain-qualcomm-inference-suite is licensed under the [BSD-3-clause License](https://spdx.org/licenses/BSD-3-Clause.html). See [LICENSE](LICENSE) for the full license text.