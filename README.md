<p align="center">
  <img width="75%" alt="smollest logo" src="assets/logo2.svg" />
</p>

The `smollest` Python library is designed to help you get the most out of your local coding model. `smollest` takes a local model like `` or `` that will run and automatically quantizes it further in the background based on your existing coding agent traces, so that the model takes even less RAM and you get more tokens/second out of it.

The basic intuition is a model like `` can program pretty well in Rust, Haskell, or Python, and understands instructions in English, German, and Arabic. But if you are always instructing it in English and writing Python & TypeScript 99% of the time, then you should be able to squeeze (quantize) the model further to get more performance for _your_ coding use cases.

*How it works (technically)*

The ideas behind `smollest` are similar to those used e.g. by Unsloth and others to do dynamic quantization. When Unsloth does dynamic quantization, it takes a large model like `` and tests it on a private dataset with millions of samples and reduces the quantization of those layers that have the least variation on  

*What if my coding habits change?*

This is a great question

## Installation

Since `smollest` is just a Python backg

## Roadmap

- Can we do further optimization via automatic kernel optimization on a user's machine?
- Can we do other kinds of optimizations, e.g. harness optimization