from pyspark.sql import SparkSession

# Crear la sesi..n de Spark                                                                                                                                                                                                                   
spark = SparkSession.builder \
    .appName("LeerCSVdeHDFS") \
    .getOrCreate()

# Especifica la ruta del archivo en HDFS                                                                                                                                                                                                      
hdfs_file_path = "hdfs://namenode:9000/input/OnlineRetail.csv"

# Leer el archivo CSV desde HDFS                                                                                                                                                                                                              
df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(hdfs_file_path)

# Mostrar la estructura del DataFrame                                                                                                                                                                                                         
df.printSchema()

# Mostrar una muestra de los datos                                                                                                                                                                                                            
df.show(5)

from pyspark.sql import functions as F

# Convertir 'InvoiceDate' a tipo fecha utilizando el formato adecuado (M/d/yyyy H:mm)                                                                                                                                                         
df = df.withColumn("InvoiceDate", F.to_timestamp("InvoiceDate", "M/d/yyyy H:mm"))

# Definir la fecha de referencia                                                                                                                                                                                                              
reference_date = F.lit("2011-12-10")

# Calcular la m..trica Recency                                                                                                                                                                                                                
df_recency = df.groupBy("CustomerID") \
    .agg(F.datediff(reference_date, F.max("InvoiceDate")).alias("Recency"))

# Calcular la m..trica Frequency (cantidad de compras por cliente)                                                                                                                                                                            
df_frequency = df.groupBy("CustomerID") \
    .agg(F.countDistinct("InvoiceNo").alias("Frequency"))

# Calcular la m..trica Monetary (suma total gastada por cliente), redondeando a 2 decimales                                                                                                                                                   
df = df.withColumn("TotalAmount", df["Quantity"] * df["UnitPrice"])
df_monetary = df.groupBy("CustomerID") \
    .agg(F.round(F.sum("TotalAmount"), 2).alias("Monetary"))

# Unir las tres m..tricas en un solo DataFrame RFM                                                                                                                                                                                            
df_rfm = df_recency.join(df_frequency, "CustomerID").join(df_monetary, "CustomerID")

# Mostrar el resultado                                                                                                                                                                                                                        
df_rfm.show(5)

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# 1. Seleccionar las columnas para K-means (Recency, Frequency, Monetary)                                                                                                                                                                     
assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="RFM_features")
df_rfm_features = assembler.transform(df_rfm)

# 2. Estandarizar los datos                                                                                                                                                                                                                   
scaler = StandardScaler(inputCol="RFM_features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(df_rfm_features)
df_rfm_scaled = scaler_model.transform(df_rfm_features)

# 3. Aplicar K-means                                                                                                                                                                                                                          
kmeans = KMeans(featuresCol="scaled_features", k=4)  # Ajusta 'k' seg..n el n..mero de clusters que quieras                                                                                                                                   
model = kmeans.fit(df_rfm_scaled)

# 4. Hacer predicciones (asignar cada cliente a un cluster)                                                                                                                                                                                   
df_rfm_clustered = model.transform(df_rfm_scaled)

# Mostrar los resultados con los clusters asignados                                                                                                                                                                                           
df_rfm_clustered.select("CustomerID", "Recency", "Frequency", "Monetary", "prediction").show(10)

# Evaluar el modelo                                                                                                                                                                                                                           
evaluator = ClusteringEvaluator(featuresCol="scaled_features")
silhouette = evaluator.evaluate(df_rfm_clustered)
print(f"Silhouette with squared euclidean distance: {silhouette}") 