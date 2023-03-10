from sklearn.model_selection import GridSearchCV
import numpy as np


def crosvalidation_DT(model, y_validation, x_validation, param_grid):
    
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(x_validation, y_validation)
    
    print("\n\nResultados de Crossvalidation:")
    print("Mejores parametros: ", grid.best_params_)
    print("Mejor score: {:.2f}".format(grid.best_score_))
    
    return grid.best_params_