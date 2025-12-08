# ICL-Attack-on-Classification
ICL attack on classification task testing with language models. 

To reproduce,
1. Download / clone repository
2. Create virtual environment, activate, pip install -r requirements.txt
3. Download Qwen3/Qwen3-0.6B from huggingface or whatever model you want to test
4. Create your prompts, can hard code in or read in from `prompts.txt` 
5. Setup model path in model.py (can search project files for `# YOUR MODEL HERE !`)
6. Run using `run_experiment.py`