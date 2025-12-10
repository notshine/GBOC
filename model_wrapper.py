def run_GBOC(data_train, data_test, win_size=10, lr=1e-3, 
        hidden_dim=32, num_layers=1, alpha=0.9, dis_mode='ball', epochs=30):
    from models.GBOC import GBOC
    clf = GBOC(win_size=win_size, feats=data_test.shape[1], lr=lr, 
        hidden_dim=hidden_dim, num_layers=num_layers, alpha=alpha, dis_mode=dis_mode, epochs=epochs)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()




