import experiment 

def main():
    ensemble_models = [model(experiment.ENSEMBLE_SIZE) for model in experiment.ENSEMBLE_MODEL_LIST]

    for model in ensemble_models:
         experiment.measure_adversarial_transfer(model, 1, save_images=True)

if __name__ == '__main__':
    main()
