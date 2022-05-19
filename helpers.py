from tkinter import Y
import pandas as pd
import numpy as np

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols, num_but_cat


# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


# Bu fonsksiyon eksik değerlerin median veya mean ile doldurulmasını sağlar
def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if
                         data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                      axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def data_preb(dataframe, num_method="median", cat_length=20, target="SalePrice"):
    print("Veri ön hazırlama başladı...")
    cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(dataframe)

    for col in num_cols:
        if col != "SalePrice":
            replace_with_thresholds(dataframe, col)

    # Bazı değişkenlerdeki boş değerler evin o özelliğe sahip olmadığını ifade etmektedir
    no_cols = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
               "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

    # Kolonlardaki boşlukların "No" ifadesi ile doldurulması
    for col in no_cols:
        dataframe[col].fillna("No", inplace=True)

    dataframe = quick_missing_imp(dataframe, num_method, cat_length)

    dataframe["ExterCond"] = np.where(dataframe.ExterCond.isin(["Fa", "Po"]), "FaPo", dataframe["ExterCond"])
    dataframe["ExterCond"] = np.where(dataframe.ExterCond.isin(["Ex", "Gd"]), "Ex", dataframe["ExterCond"])

    dataframe["LotShape"] = np.where(dataframe.LotShape.isin(["IR1", "IR2", "IR3"]), "IR", dataframe["LotShape"])

    dataframe["GarageQual"] = np.where(dataframe.GarageQual.isin(["Fa", "Po"]), "FaPo", dataframe["GarageQual"])
    dataframe["GarageQual"] = np.where(dataframe.GarageQual.isin(["Ex", "Gd"]), "ExGd", dataframe["GarageQual"])
    dataframe["GarageQual"] = np.where(dataframe.GarageQual.isin(["ExGd", "TA"]), "ExGd", dataframe["GarageQual"])

    dataframe["BsmtFinType2"] = np.where(dataframe.BsmtFinType2.isin(["GLQ", "ALQ"]), "RareExcellent", dataframe["BsmtFinType2"])
    dataframe["BsmtFinType2"] = np.where(dataframe.BsmtFinType2.isin(["BLQ", "LwQ", "Rec"]), "RareGood", dataframe["BsmtFinType2"])

    # Nadir sınıfların tespit edilmesi
    rare_encoder(dataframe, 0.01)

    dataframe["NEW_1st*GrLiv"]=(dataframe["1stFlrSF"]*dataframe["GrLivArea"])
    dataframe["NEW_Garage*GrLiv"]=(dataframe["GarageArea"]*dataframe["GrLivArea"])
    dataframe["NEW_TotalQual"] = dataframe[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                          "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis = 1)
    dataframe["NEW_TotalGarageQual"] = dataframe[["GarageQual", "GarageCond"]].sum(axis = 1)
    dataframe["NEW_Overall"] = dataframe[["OverallQual", "OverallCond"]].sum(axis = 1)
    dataframe["NEW_Exter"] = dataframe[["ExterQual", "ExterCond"]].sum(axis = 1)
    dataframe["NEW_Qual"] = dataframe[["OverallQual", "ExterQual", "GarageQual", "Fence", "BsmtFinType1", "BsmtFinType2", "KitchenQual", "FireplaceQu"]].sum(axis = 1)
    dataframe["NEW_Cond"] = dataframe[["OverallCond", "ExterCond", "GarageCond", "BsmtCond", "HeatingQC", "Functional"]].sum(axis = 1)
    # Total Floor
    dataframe["NEW_TotalFlrSF"] = dataframe["1stFlrSF"] + dataframe["2ndFlrSF"]
    # Total Finished Basement Area
    dataframe["NEW_TotalBsmtFin"] = dataframe.BsmtFinSF1+dataframe.BsmtFinSF2
    # Porch Area
    dataframe["NEW_PorchArea"] = dataframe.OpenPorchSF + dataframe.EnclosedPorch + dataframe.ScreenPorch + dataframe["3SsnPorch"] + dataframe.WoodDeckSF
    # Total House Area
    dataframe["NEW_TotalHouseArea"] = dataframe.NEW_TotalFlrSF + dataframe.TotalBsmtSF
    dataframe["NEW_TotalSqFeet"] = dataframe.GrLivArea + dataframe.TotalBsmtSF
    # Lot Ratio
    dataframe["NEW_LotRatio"] = dataframe.GrLivArea / dataframe.LotArea
    dataframe["NEW_RatioArea"] = dataframe.NEW_TotalHouseArea / dataframe.LotArea
    dataframe["NEW_GarageLotRatio"] = dataframe.GarageArea / dataframe.LotArea
    # MasVnrArea
    dataframe["NEW_MasVnrRatio"] = dataframe.MasVnrArea / dataframe.NEW_TotalHouseArea
    # Dif Area
    #dataframe["NEW_DifArea"] = (dataframe.LotArea - dataframe["1stFlrSF"] - dataframe.GarageArea - dataframe.NEW_PorchArea - dataframe.WoodDeckSF)
    # LowQualFinSF
    dataframe["NEW_LowQualFinSFRatio"] = dataframe.LowQualFinSF / dataframe.NEW_TotalHouseArea
    dataframe["NEW_OverallGrade"] = dataframe["OverallQual"] * dataframe["OverallCond"]
    # Overall kitchen score
    dataframe["NEW_KitchenScore"] = dataframe["KitchenAbvGr"] * dataframe["KitchenQual"]
    # Overall fireplace score
    dataframe["NEW_FireplaceScore"] = dataframe["Fireplaces"] * dataframe["FireplaceQu"]
    dataframe["NEW_Restoration"] = dataframe.YearRemodAdd - dataframe.YearBuilt
    dataframe["NEW_HouseAge"] = dataframe.YrSold - dataframe.YearBuilt
    dataframe["NEW_RestorationAge"] = dataframe.YrSold - dataframe.YearRemodAdd
    dataframe["NEW_GarageAge"] = dataframe.GarageYrBlt - dataframe.YearBuilt
    dataframe["NEW_GarageRestorationAge"] = np.abs(dataframe.GarageYrBlt - dataframe.YearRemodAdd)
    dataframe["NEW_GarageSold"] = dataframe.YrSold - dataframe.GarageYrBlt

    drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]

    # drop_list'teki değişkenlerin düşürülmesi
    dataframe.drop(drop_list, axis=1, inplace=True)

    cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(dataframe)

    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)
    print("Veri ön hazırlama tamamlandı. Tebrikler!")

    y = dataframe["SalePrice"]
    X = dataframe.drop(["Id", "SalePrice"], axis=1)

    return X, y

