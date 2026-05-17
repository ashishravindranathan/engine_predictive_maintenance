---
title: Engine Predictive Maintenance
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Engine Predictive Maintenance Space

This Space:
- loads the saved model from the Hugging Face Model Hub
- gets user inputs
- saves the inputs into a dataframe
- predicts engine condition
- runs inside a Docker container
