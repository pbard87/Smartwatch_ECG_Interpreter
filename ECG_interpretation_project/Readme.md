## Using the MedGemma Interpreter ##

### Setting up the Environemnt ###

To create a new conda envirionment and install necessary packages run:  
`conda env create -f environment.yml -n ecg_interp_test`  
  
To install necessary packages in an existing environment run:  
`conda env update -n myenv --file environment.yaml`  
  
Additional packages may need to be installed with pip install -r `requirements.txt`  
  
### Running the Model  
On a Cuda compatible system run 
`python3 ECG_interpretation_with_medgemma_agentic.py `

If a medgemma token is necessary follow the instructions here: https://huggingface.co/google/medgemma-1.5-4b-it to set one up.
