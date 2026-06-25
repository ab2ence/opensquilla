# Fusion Experiment Settings

Date: 2026-06-25

This document records the G-series experiment group settings for the current
Fusion Assist evaluation. The judge model is fixed to `z-ai/glm-5.2` for all
groups.

| 组 | Fusion 层模型 | Action model | Judge model |
|---|---|---|---|
| G1 | `deepseek/deepseek-v4-pro`<br>`z-ai/glm-5.2`<br>`moonshotai/kimi-k2.7-code` | `z-ai/glm-5.2` | `z-ai/glm-5.2` |
| G2 | `z-ai/glm-5.2`<br>`qwen/qwen3.7-plus`<br>`moonshotai/kimi-k2.6` | `z-ai/glm-5.2` | `z-ai/glm-5.2` |
| G3 | `deepseek/deepseek-v4-pro`<br>`z-ai/glm-5.2`<br>`google/gemini-3-flash-preview` | `z-ai/glm-5.2` | `z-ai/glm-5.2` |
| G4 | `deepseek/deepseek-v4-pro`<br>`z-ai/glm-5.2`<br>`google/gemini-3-flash-preview` | `google/gemini-3-flash-preview` | `z-ai/glm-5.2` |
| G5 | `deepseek/deepseek-v4-pro`<br>`z-ai/glm-5.2`<br>`google/gemini-3-flash-preview` | `anthropic/claude-opus-4.8` | `z-ai/glm-5.2` |
| G6 | `deepseek/deepseek-v4-pro`<br>`z-ai/glm-5.2`<br>`google/gemini-3-flash-preview` | `openai/gpt-5.5` | `z-ai/glm-5.2` |
| G7 | `deepseek/deepseek-v4-pro`<br>`z-ai/glm-5.2` | `z-ai/glm-5.2` | `z-ai/glm-5.2` |
| G8 | `deepseek/deepseek-v4-pro`<br>`z-ai/glm-5.2`<br>`google/gemini-3-flash-preview`<br>`qwen/qwen3.7-plus` | `z-ai/glm-5.2` | `z-ai/glm-5.2` |
