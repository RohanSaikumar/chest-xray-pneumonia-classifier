# Evaluation logic
def evaluate(model, test_gen):
    loss, accuracy = model.evaluate(test_gen, verbose=2)
    return loss, accuracy
