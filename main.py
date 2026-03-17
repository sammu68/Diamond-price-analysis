from analysis import load_data, basic_analysis
from model import train_model
def main():
    print("Loading data...")
    df = load_data()
    print("\nAnalyzing data...")
    basic_analysis(df)
    print("\nTraining model...")
    train_model(df)
if __name__ == "__main__":
    main()