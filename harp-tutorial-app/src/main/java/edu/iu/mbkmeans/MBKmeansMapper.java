package edu.iu.mbkmeans;

import edu.iu.harp.example.DoubleArrPlus;
import edu.iu.harp.partition.Partition;
import edu.iu.harp.partition.Table;
import edu.iu.harp.resource.DoubleArray;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.CollectiveMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Created by Ethan on 4/10/17.
 */
public class MBKmeansMapper  extends CollectiveMapper<String, String, LongWritable, Text> {
    private static final Logger LOG = LoggerFactory.getLogger(MBKmeansMapper.class);
    int dimension;
    int numOfCentroids;
    int batchSize;
    int localBatchSize;
    int iterations;
    int numMapTasks;
    Random rand;
    Configuration conf;
    boolean DEBUG;
    public void setup(Context context) {
        conf = context.getConfiguration();
        dimension = conf.getInt(MBKmeansConstants.DIMENSION, 1);
        numOfCentroids = conf.getInt(MBKmeansConstants.K, 1);
        batchSize = conf.getInt(MBKmeansConstants.BATCH_SIZE, 0);
        iterations = conf.getInt(MBKmeansConstants.NUM_ITERATONS, 1);
        numMapTasks = conf.getInt(MBKmeansConstants.NUM_MAP_TASKS, 1);

        // calculate local batch size
        int mod = batchSize % numMapTasks;
        localBatchSize = batchSize / numMapTasks;
        if (mod != 0){
            if (this.getSelfID() < mod){
                ++localBatchSize;
            }
        }

        rand = new Random();
        DEBUG = true;
    }

    /**
     * mapCollective task
     * @param reader
     * @param context
     * @throws IOException
     * @throws InterruptedException
     */
    public void mapCollective(KeyValReader reader, Context context) throws IOException, InterruptedException {
        List<String> inputFiles = new ArrayList<String>();
        while (reader.nextKeyValue()) {
            String key = reader.getCurrentKey();
            String value = reader.getCurrentValue();
            LOG.info("Key: " + key + ", Value: " + value);
            inputFiles.add(value);
        }

        //1. load graph
        //use primitive types to save memory usage
        List<double[]> dataList = loadData(inputFiles, this.dimension, conf);
        LOG.info("Loaded "+ dataList.size()+" data points in total");
        //2. generate initial centroids
        Table<DoubleArray> cenTable = new Table<>(0, new DoubleArrPlus());

        if( this.isMaster()){
            initCentroids(cenTable, dataList, dimension);
        }

        //print table for testing
        if(DEBUG) printTable(cenTable);
        /*
        //3. do iterations
        for(int iter = 0; iter < iterations; ++iter){

            // 3.1 randomly pick localBatchSize of dataset
            int[] dataSampleIds = new int[localBatchSize];
            for( int i = 0; i < localBatchSize; i++) {
                dataSampleIds[i] =  rand.nextInt( dataList.size() ) ;
            }

            int[] cachedCentroids = new int[localBatchSize];
            double[] cachedDistance = new double[localBatchSize];
            //initialize cachedDistance
            for(int j = 0; j < localBatchSize; ++j) cachedDistance[j] = Double.MAX_VALUE;

            // rotate and for-loop to calculate the nearest centroids for each x in dataSamples.
            int rotationNo = 0;
            do {//do rotation
                LOG.info("Updating the nearest centroid for all data samples");

                for( int i = 0; i < localBatchSize; i++) {
                    updateNearestCentroid(i, dataSampleIds,cachedCentroids, cachedDistance, cenTable, dataList);
                }

                LOG.info("[BEGIN] rotation " + rotationNo);
                rotate ("MBKmeans-nearestCentroids","rotation"+rotationNo, cenTable, null);
                LOG.info("[END] rotation " + rotationNo);
                ++rotationNo;
            }while(rotationNo < numMapTasks);
            LOG.info("[DONE] Find nearest centroids");

            rotationNo = 0;
            Map<Integer, List<Integer>> map = invertedIndex(dataSampleIds, cachedCentroids);

            //print the map size
            if(DEBUG) {
                System.out.println("output the inverted index");
                for (Map.Entry<Integer, List<Integer>> entry : map.entrySet()) {
                    System.out.println(entry.getKey() + "\t" + entry.getValue().size());
                }
            }

            do {//do rotation
                LOG.info("Updating centroids in cenTable");
                for( Partition<DoubleArray> partition: cenTable.getPartitions()){
                    int cenId = partition.id();
                    double[] centroid = partition.get().get();
                    if( map.containsKey(cenId)) {
                        List data = map.get(cenId);
                        for( int j = 0; j < data.size(); j++){
                            ++centroid[dimension];                // Update per-center counts
                            double eta = 1 / centroid[dimension]; //per-center learning rate
                            gradientStep(centroid, dataList.get(dataSampleIds[j]), eta);             // Take gradient step
                        }
                    }
                }

                LOG.info("[BEGIN] rotation " + rotationNo);
                rotate ("MBKmeans-updateCentroids","rotation"+rotationNo, cenTable, null);
                LOG.info("[END] rotation " + rotationNo);
                ++rotationNo;
            }while(rotationNo < numMapTasks);

            LOG.info("[DONE] iteration " + iter);
        }

        allgather("MBKmeans-gatherCentroids", "allgather", cenTable);
        if( this.isMaster()){
            outputCentroids(cenTable,  conf,   context);
        }
        */
    }

    /**
     * print Table
     * @param cenTable
     */
    private void printTable(Table<DoubleArray> cenTable){
        for( Partition<DoubleArray> ap: cenTable.getPartitions()){

            double res[] = ap.get().get();
            System.out.print("ID: "+ap.id() + ":");
            for(int i=0; i<res.length;i++)
                System.out.print(res[i]+"\t");
            System.out.println();
        }
    }

    private void outputCentroids(Table<DoubleArray> cenTable, Configuration conf, Context context){
        String output="";
        for( Partition<DoubleArray> ap: cenTable.getPartitions()){
            output += "ID: "+ ap.id() + "\t";
	    double res[] = ap.get().get();
            for(int i = 0; i < dimension;i++)
                output+= res[i]+"\t";
            output+="\n";
        }
        try {
            context.write(null, new Text(output));
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * Take the gradient step
     * @param centroid
     * @param data
     * @param eta
     */
    private void gradientStep(double[] centroid, double[] data, double eta){
        for( int i = 0; i < data.length; ++i){
            centroid[i] = ( 1 - eta) * centroid[i] + eta * data[i];
        }
    }

    /**
     * create invertedIndex so that cenId maps to the data list in which all the data belongs to the cenId.
     * @param dataSamples
     * @param cachedCentroids
     * @return
     */
    private Map<Integer, List<Integer>> invertedIndex(int[] dataSamples, int[]cachedCentroids){
        Map<Integer, List<Integer>> invertIndex = new HashMap();

        for(int i = 0; i< dataSamples.length; i++){
            if( invertIndex.containsKey( cachedCentroids[i])){
                List<Integer> list = invertIndex.get(cachedCentroids[i]);
                list.add( dataSamples[i]);
            }else{
                List<Integer> list = new ArrayList();
                list.add( dataSamples[i] );
                invertIndex.put(cachedCentroids[i], list);
            }
        }
        return invertIndex;
    }

    /**
     * find the nearest centroid for data dataSamples[i]
     * @param i
     * @param cachedCentroids
     * @param cachedDistance
     * @param cenTable
     * @return
     */
    private void updateNearestCentroid(int i, int[] dataSamples, int[] cachedCentroids,
                                       double[] cachedDistance, Table<DoubleArray> cenTable, List<double[]> dataList){
        int dataLocalId = dataSamples[i];
        double[] data =  dataList.get(dataLocalId);
        for( Partition<DoubleArray> partition: cenTable.getPartitions()){
            int cenId = partition.id();

            double dist = distance(data, partition.get().get(), this.dimension);

            if(DEBUG) System.out.println("calculate cenId= "+cenId+"; dist = " + dist + "; cached dist: "+ cachedDistance[i]);
            if ( dist < cachedDistance[i]){
                cachedCentroids[i] = cenId;
                cachedDistance[i] = dist;
            }
            if(DEBUG) System.out.println("nearest centroid id is "+ cachedCentroids[i]+"; distance: "+ cachedDistance[i]);
        }
        if(DEBUG) System.out.println("nearest centroid id is "+ cachedCentroids[i]+"; distance: "+ cachedDistance[i]);
    }
    private double distance(double[] x, double[] y, int dimension ){
        double dist = 0;
        for( int i = 0; i < dimension; i++){
            dist += Math.pow(x[i] - y[i], 2);
        }

        return (double) Math.sqrt(dist);
    }

    /**
     * Initialize centroids
     * @param cenTable
     * @param dataList
     */
    private void initCentroids(Table<DoubleArray> cenTable, List<double[]> dataList, int dimension){
        //randomly pick from dataset.
        int centroidId = 0;
        for( int i = 0; i < numOfCentroids; i++){
            int c = rand.nextInt(dataList.size());
            double[] copy = Arrays.copyOf(dataList.get(c), dimension+1);
            Partition<DoubleArray> ap = new Partition<DoubleArray>(centroidId, new DoubleArray(copy, 0, dimension+1));
            cenTable.addPartition(ap);
            LOG.info("centroid id " + centroidId +" added to the cenTable on node " + this.getSelfID());
            ++centroidId;
        }
        LOG.info(cenTable.getNumPartitions() + " centroids added to the cenTable on node " + this.getSelfID());
    }

    /**
     * Load data
     * @param fileNames input files
     * @param dimension dimension of the data
     * @param conf Hadoop Configuration
     * @return list of double[]
     * @throws IOException
     */
    private List<double[]> loadData(List<String> fileNames, int dimension, Configuration conf)
            throws IOException{
        List<double[]> dataList  = new ArrayList<double[]>();
        for(String filename: fileNames){
            FileSystem fs = FileSystem.get(conf);
            Path dPath = new Path(filename);
            FSDataInputStream in = fs.open(dPath);
            BufferedReader br = new BufferedReader( new InputStreamReader(in));
            String line="";
            String[] vector=null;
            while((line = br.readLine()) != null){
                vector = line.split("\\s+");

                if(vector.length != dimension  ){
                    LOG.error("Errors while loading data.");
                    System.exit(-1);
                }else{
                    double[] aDataPoint = new double[dimension];
                    for(int i = 0; i < dimension; i++){
                        aDataPoint[i] = Double.parseDouble(vector[i]);
                    }
                    dataList.add(aDataPoint);
                }
            }
        }
        return dataList;
    }



}
