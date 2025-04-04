# DS4300-Practical-2

### How to Run MiniLM scripts
- Run preprocess_text.py to generate text in the data folder.
- Run miniLM_comparison_script.py to generate responses based of different vector DBs, chunking strategies, and LLM choices. May need to change location of redis port, and data directories to point to the correct chunked texts.
- Similar to miniLM_comparison_script.py, will need to ensure redis and data directories are updated to run miniLM_interactive.py
