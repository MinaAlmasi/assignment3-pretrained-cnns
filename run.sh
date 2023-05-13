# activate virtual environment 
source ./env/bin/activate

# run classification
echo -e "[INFO:] Running classification pipeline ..."
python src/classify_CNN.py

# deactivate env 
deactivate

# happy user msg!! 
echo -e "[INFO:] Pipeline complete! Model information stored!"