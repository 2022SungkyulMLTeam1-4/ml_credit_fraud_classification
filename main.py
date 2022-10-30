import classification

if __name__ == '__main__':
    model = classification.Model('credit_model', 'creditcard.csv')
    model.train()
    print(model.evaluate())
