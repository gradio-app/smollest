<p align="center">
  <img width="75%" alt="smollest logo" src="assets/logo2.svg" />
</p>

The `smollest` Python library is designed to help you get the most out of your local coding model. `smollest` takes a local model like `` or `` that will run and automatically quantizes it further in the background based on your existing coding agent traces, so that the model takes even less RAM and you get more tokens/second out of it.

*How it works (intuitively)*

The basic intuition is a model like `` can program pretty well in Rust, Haskell, or Python, and understands instructions in English, German, and Arabic. But if you are always instructing it in English and writing Python & TypeScript 99% of the time, then you should be able to squeeze (quantize) the model further to get more performance for _your_ coding use cases.

*How it works (technically)*

The ideas behind `smollest` are similar to those used e.g. by Unsloth and others to do dynamic quantization. When Unsloth does dynamic quantization, it takes a large model like `` and tests it on a private dataset with millions of samples and reduces the quantization of those layers that have the least variation on this dataset. We do the same thing, but on your specific data (specifically, your previous Codex/Claude/Pi agent traces and this happens locally, no data leaves your machine). In the end, you have a **hyper-specific quantized coding model**. For more of the gnarly details, see the [technical details]() doc.

*What if my coding habits change?*

This is a great question. The great thing about quantization is that we can see if your queries are out-of-distribution of the quantization. If that happens, `smollest` currently prints a warning letting you know. It also requantizes every week by default on the most recent traces, although you can rerun it at anytime using `smollest smollify` (see below). 

*How long does quantization take?*

It depends on how many agent traces and what machine you are using. On a recent Macbook Pro and with the default 1,000 most recent traces, it takes approximately 30-60 minutes. You can see the progress at anytime using `smollest view` (see below).

## Get Started

`smollest` requires Python 3.10+, then install it with:

```
uv pip install --upgrade smollest
```

Then simply run in your terminal:

```
smollest setup
```

This will prompt you to choose a local model that will be the default target for optimization. If you already have a local model that you use, you can select that, otherwise you ca

```
smollest smollify
```

This will start the quantization process. You can monitor the progress of the quantization by running in your terminal:

```
smollest view
```

When the model finishes quantizing, it will be published under your Hugging Face account (as a private model by default). You can modify this as well as the parameters of the quantization process using various [command-line flags]().

## Roadmap / questions

- Can we do further optimization via automatic kernel optimization on a user's machine?
- Can we run the optimizations using HF Jobs, or a user's existing Codex subscription using ACP?
- Can we do other kinds of optimizations, e.g. harness optimization?