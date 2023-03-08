import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def data_cleaning():
    df = pd.read_csv('dataRanked.csv')
    #print(df['blueWins'].value_counts())
    
    
    # Na's en las columnas
    #print("Verificando que no haya Na's en las columnas")
    #print(df.isna().any())
    
    # No existen Na's en las columnas
    
    categoricas=df['blueFirstBlood']
    
    cuantitativas= df.drop(['blueFirstBlood','gameId', 'blueWins'], axis=1)
    

    a=df[(df['blueFirstBlood'] == 1) & (df['blueWins'] == 1)].shape[0]



    #print("Victorias del equipo azul cuando hacen FirstBlood",a/df.shape[0])
    
    for column in cuantitativas.columns:
        #print("Analisis de la columna: ", column)
        media = df.groupby('blueWins')[column].describe()
        #print(media,end="\n\n")

    
    corr_matrix = cuantitativas.corr()
    

    plt.xticks(rotation=45, ha='right')
    
    sns.heatmap(corr_matrix,cmap='coolwarm')
    #plt.show(block = True)
    
    
    directa_prop = corr_matrix.mask((corr_matrix <= 0.75) | (corr_matrix >= 1))

    inver_prop = corr_matrix.mask(corr_matrix >= -0.75) 
    
    
    inver_prop = inver_prop.stack()
    directa_prop = directa_prop.stack()
    
    
    #print("Correlaciones inversas mayores a -0.75")
    for col_pair,corr_val in inver_prop.items():
        col1,col2 = col_pair
        #print(f"{col1} - {col2}: {corr_val}")
    
    
    #print("\n Correlaciones inversas mayores a 0.75")
    for col_pair,corr_val in directa_prop.items():
        col1,col2 = col_pair
        #print(f"{col1} - {col2}: {corr_val}")
    
    

    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    
    
    #Aun se pueden quitar más 
    variables_sifnicativas = df.drop(['gameId','redGoldDiff','redExperienceDiff','redGoldPerMin','blueGoldPerMin',
                                      'redCSPerMin','blueCSPerMin','blueAssists','redAssists','blueDeaths',
                                      'redDeaths','blueFirstBlood',
                                      'redEliteMonsters','blueEliteMonsters','blueDragons','redDragons',
                                      'redTotalMinionsKilled','blueTotalMinionsKilled','redAvgLevel','blueAvgLevel'
                                
                                    ], axis=1)
    """
    Tras el analisis de correlación de las variables, pudimos observar que hay muchas varaibles
    con una relación inversa ya que se esta descrbiendo una caracteristica desde el equipo 
    rojo y otra desde el equipo azul. Por lo que decidimos eliminar una de estas variables especificamente
    las que describen las caracteristicas desde el punto de vista del equipo rojo.
    
    A su vez existen variables demasiado correlacionadas, por lo que decidimos eliminar aquellas que dieran la mis
    información que otras variables. Por ejemplo blueGoldPerMin y redGoldPerMin y blueTotalGold y redTotalGold a mayor 
    cantidad de oro por minuto mayor sera la cantidad total. Entonces para no generar overfitting decidimos eliminarlas
    """
    
    return variables_sifnicativas
    

