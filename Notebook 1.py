#!/usr/bin/env python
# coding: utf-8

# ## Notebook 1
# 
# 
# 

# In[31]:


get_ipython().run_cell_magic('pyspark', '', 'blob_account_name = "carlostorres"\nblob_container_name = "taxidata"\nfrom pyspark.sql import SparkSession\n\nsc = SparkSession.builder.getOrCreate()\ntoken_library = sc._jvm.com.microsoft.azure.synapse.tokenlibrary.TokenLibrary\nblob_sas_token = token_library.getConnectionString("BlobdeTarea")\n\nspark.conf.set(\n    \'fs.azure.sas.%s.%s.blob.core.windows.net\' % (blob_container_name, blob_account_name),\n    blob_sas_token)\ndf = spark.read.load(\'wasbs://taxidata@carlostorres.blob.core.windows.net/taxis.csv\', format=\'csv\',\nheader=True\n)\ndisplay(df.limit(10))\n')


# In[32]:


# Azure storage access info
blob_account_name = "azureopendatastorage"
blob_container_name = "nyctlc"
blob_relative_path = "yellow"
blob_sas_token = r""

# Allow SPARK to read from Blob remotely
wasbs_path = 'wasbs://%s@%s.blob.core.windows.net/%s' % (blob_container_name, blob_account_name, blob_relative_path)
spark.conf.set(
  'fs.azure.sas.%s.%s.blob.core.windows.net' % (blob_container_name, blob_account_name),
  blob_sas_token)
print('Remote blob path: ' + wasbs_path)

# SPARK read parquet, note that it won't load any data yet by now
nyc_tlc_spark_df = spark.read.parquet(wasbs_path)
print('Register the DataFrame as a SQL temporary view: source')
nyc_tlc_spark_df.createOrReplaceTempView('source')

# Display top 10 rows
print('Displaying top 10 rows: ')
display(spark.sql('SELECT * FROM source LIMIT 10'))


# In[33]:


sc


# In[34]:


#### DO NOT CHANGE ANYTHING IN THIS CELL ####

from pyspark.sql.functions import col

def load_data(size='small'):
    # Loads the data for this question. Do not change this function.
    # This function should only be called with the parameter 'small' or 'large'

    if size != 'small' and size != 'large':
        print("Invalid size parameter provided. Use only 'small' or 'large'.")
        return

    input_bucket = "s3://lab11-janedoe3"

    # Load Trip Data
    trip_path = '/'+size+'/yellow_tripdata*'
    trips = spark.read.csv(input_bucket + trip_path, header=True, inferSchema=True)
    print("Trip Count: ",trips.count()) # Prints # of trips (# of records, as each record is one trip)

    # Load Lookup Data
    lookup_path = '/'+size+'/taxi*'
    lookup = spark.read.csv(input_bucket + lookup_path, header=True, inferSchema=True)

    return trips, lookup

def main(size, bucket):
    # Runs your functions implemented above.

    print(user())
    trips, lookup = load_data(size=size)
    trips = long_trips(trips)
    mtrips = manhattan_trips(trips, lookup)
    wp = weighted_profit(trips, mtrips)
    final = final_output(wp,lookup)

    # Outputs the results for you to visually see
    final.show()

    # Writes out as a CSV to your bucket.
    final.write.csv(bucket)


# In[35]:


def long_trips(trips = 2):
    """
    Filtra las filas del DataFrame donde tripDistance es mayor que un valor dado (2)
    """
    return nyc_tlc_spark_df.filter(col("tripDistance") > trips)


# In[36]:


from pyspark.sql.functions import col, desc, sum as spark_sum

def manhattan_trips(nyc_tlc_spark_df, df):
    
    # Filtrar LocationIDs de Manhattan
    manhattan_ids = df.filter(col("Borough") == "Manhattan").select("LocationID").distinct()
    
    # Filtrar viajes con DOLocationID en Manhattan
    manhattan_trips = nyc_tlc_spark_df.join(
        manhattan_ids,
        nyc_tlc_spark_df["doLocationId"] == manhattan_ids["LocationID"],
        "inner"
    )
    
    # Calcular el top 20 basado en la suma de passenger_count
    top_manhattan_trips = (
        manhattan_trips.groupBy("doLocationId")
        .agg(spark_sum("passengerCount").alias("total_passengers"))
        .orderBy(desc("total_passengers"))
        .limit(20)
    )
    
    return top_manhattan_trips


# In[37]:


trips = long_trips(2)
mtrips = manhattan_trips(nyc_tlc_spark_df, df)


# In[38]:


def weighted_profit(trips, mtrips):
    # Crear una lista con los 20 destinos más populares
    top20_destinations = (
        mtrips.select("DOLocationID")
        .rdd.map(lambda row: row.DOLocationID)
        .collect()
    )

    # Filtrar los viajes de más de 2 millas
    filtered_trips = trips.filter(F.col("tripDistance") > 2)

    # Crear una nueva columna para indicar si el destino está en los 20 más populares
    trips_with_flag = filtered_trips.withColumn(
        "is_top20",
        F.when(F.col("doLocationId").isin(top20_destinations), 1).otherwise(0),
    )

    # Calcular las métricas necesarias
    result = (
        trips_with_flag.groupBy("puLocationId")
        .agg(
            F.avg("totalAmount").alias("avg_total_amount"),
            F.count("*").alias("total_trips"),
            F.sum("is_top20").alias("trips_to_top20"),
        )
        .withColumn(
            "proportion_to_top20",
            F.col("trips_to_top20") / F.col("total_trips"),
        )
        .withColumn(
            "weighted_profit",
            F.col("proportion_to_top20") * F.col("avg_total_amount"),
        )
        .select("puLocationId", "weighted_profit")
    )

    return result


# In[39]:


from pyspark.sql import functions as F
top_20_weighted_profit = weighted_profit(trips, mtrips)

# Mostrar el resultado
top_20_weighted_profit.show(20)


# In[40]:


def final_output(wp, df):
    """
    Combina el DataFrame de weighted_profit con información de Borough y Zone,
    y selecciona las 20 ubicaciones con mayor weighted_profit.

    Args:
    wp (DataFrame): Resultado de la función weighted_profit con las columnas PULocationID y weighted_profit.
    df (DataFrame): DataFrame con información de Borough y Zone.

    Returns:
    DataFrame: DataFrame con las columnas Zone, Borough, weighted_profit (top 20 por weighted_profit).
    """
    # Realizamos el join entre wp y df usando PULocationID y LocationID
    joined_df = wp.join(
        df,
        wp["PULocationID"] == df["LocationID"],  # Une por LocationID
        "inner"
    )

    # Seleccionamos las columnas necesarias
    result = (
        joined_df.select(
            F.col("Zone"),               # Columna de la zona
            F.col("Borough"),            # Columna del borough
            F.col("weighted_profit")     # Columna del profit
        )
        .orderBy(F.col("weighted_profit").desc())  # Ordenar por weighted_profit en orden descendente
        .limit(20)  # Seleccionar los 20 primeros
    )

    return result


# In[41]:


final_df = final_output(top_20_weighted_profit, df)
final_df.show()


# In[42]:


# Ejecutar la función y guardar el resultado
top_locations = final_output(top_20_weighted_profit, df)

# Exportar a un archivo CSV llamado output.csv
top_locations.write.csv("output.csv", header=True, mode="overwrite")

print("El archivo 'output.csv' se ha creado exitosamente.")

