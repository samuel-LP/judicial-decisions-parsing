from sklearn.metrics import f1_score


class Metrics():
    def __init__(self, df, y_true, y_pred):
        self.df = df
        self.y_true = y_true
        self.y_pred = y_pred

    def accuracy(self):

        good_preds = (self.df[self.y_true] == self.df[self.y_pred]).sum()
        lenght = self.df.shape[0]

        accuracy = good_preds / lenght
        return accuracy

    def f1(self):

        y_t = (self.df[self.y_true] == self.df[self.y_true]).astype(int)
        y_p = (self.df[self.y_pred] == self.df[self.y_true]).astype(int)

        f1 = f1_score(y_t, y_p)

        return f1
