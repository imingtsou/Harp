package edu.iu.minibatchkmeans;

import java.io.IOException;
import java.net.URISyntaxException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.concurrent.ExecutionException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import edu.iu.fileformat.MultiFileInputFormat;

public class KmeansMapCollective extends Configured implements Tool {

	public static void main(String[] argv) throws Exception {
		int res = ToolRunner.run(new Configuration(), new KmeansMapCollective(), argv);
		System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		if (args.length < 7) {
			System.err.println("Usage: KmeansMapCollective <numOfDataPoints> <num of Centroids> "
					+ "<size of vector> <number of map tasks> <number of iteration> <workDir> <localDir>\n");
			ToolRunner.printGenericCommandUsage(System.err);
				return -1;
		}

		int numOfDataPoints = Integer.parseInt(args[0]);
		int numCentroids = Integer.parseInt(args[1]);	
		int sizeOfVector = Integer.parseInt(args[2]);
		int numMapTasks = Integer.parseInt(args[3]);
		int numIteration = Integer.parseInt(args[4]);
		String workDir = args[5];
		String localDir = args[6];

		System.out.println( "Number of Map Tasks = "	+ numMapTasks);
		System.out.println("Len : " + args.length);
		System.out.println("Args=:");
		for(String arg: args){
			System.out.print(arg+";");
		}
		System.out.println();
		
		launch(numOfDataPoints, numCentroids, sizeOfVector, numMapTasks, numIteration, workDir, localDir);
		System.out.println("HarpKmeans Completed"); 
		return 0;
	}
	void launch(int numOfDataPoints, int numCentroids, int sizeOfVector, int numMapTasks, int numIteration, String workDir, String localDir)
			throws IOException, URISyntaxException, InterruptedException, ExecutionException, ClassNotFoundException {
		
		Configuration configuration = getConf();
		Path workDirPath = new Path(workDir);
		FileSystem fs = FileSystem.get(configuration);
		Path dataDir = new Path(workDirPath, "data");
		Path cenDir = new Path(workDirPath, "centroids");
		Path outDir = new Path(workDirPath, "out");
		if (fs.exists(outDir)) {
			fs.delete(outDir, true);
		}
		fs.mkdirs(outDir);
		
		System.out.println("Generate data.");
		Utils.generateData(numOfDataPoints, sizeOfVector, numMapTasks, fs, localDir, dataDir);
		
		int JobID = 0;
		Utils.generateInitialCentroids(numCentroids, sizeOfVector, configuration, cenDir, fs, JobID);
		
		long startTime = System.currentTimeMillis();
		
		runKMeans(numOfDataPoints,numCentroids, sizeOfVector, numIteration,
				JobID,  numMapTasks, configuration, workDirPath,
				dataDir, cenDir, outDir);
		long endTime = System.currentTimeMillis();
		System.out.println("Total K-means Execution Time: "+ (endTime - startTime));
	}
	
	
	private void runKMeans(int numOfDataPoints, int numCentroids, int vectorSize, int numIterations, 
				int JobID, int numMapTasks, Configuration configuration, 
			Path workDirPath, Path dataDir, Path cDir, Path outDir)
			throws IOException,URISyntaxException, InterruptedException,ClassNotFoundException {
			
		System.out.println("Starting Job");
		long jobSubmitTime;
		boolean jobSuccess = true;
		int jobRetryCount = 0;
		
		do {
			// ----------------------------------------------------------------------
			jobSubmitTime = System.currentTimeMillis();
			System.out.println("Start Job#" + JobID + " "+ new SimpleDateFormat("HH:mm:ss.SSS").format(Calendar.getInstance().getTime()));
			
			Job kmeansJob = configureKMeansJob(numOfDataPoints,	numCentroids, vectorSize, numMapTasks,
					configuration, workDirPath, dataDir,cDir, outDir, JobID, numIterations);
			
			System.out.println("| Job#"+ JobID+ " configure in "+ (System.currentTimeMillis() - jobSubmitTime)+ " miliseconds |");
			
			// ----------------------------------------------------------
			jobSuccess =kmeansJob.waitForCompletion(true);
			
			System.out.println("end Jod#" + JobID + " "
						+ new SimpleDateFormat("HH:mm:ss.SSS")
						.format(Calendar.getInstance().getTime()));
			System.out.println("| Job#"+ JobID + " Finished in "
					+ (System.currentTimeMillis() - jobSubmitTime)
					+ " miliseconds |");
		
			// ---------------------------------------------------------
			if (!jobSuccess) {
				System.out.println("KMeans Job failed. Job ID:"+ JobID);
				jobRetryCount++;
				if (jobRetryCount == 3) {
					break;
				}
			}else{
				break;
			}
		} while (true);
	}
	
	private Job configureKMeansJob(int numOfDataPoints, int numCentroids, int vectorSize, 
			int numMapTasks, Configuration configuration,Path workDirPath, Path dataDir, Path cDir,
			Path outDir, int jobID, int numIterations) throws IOException, URISyntaxException {
			
		Job job = Job.getInstance(configuration, "kmeans_job_"+ jobID);
		Configuration jobConfig = job.getConfiguration();
		Path jobOutDir = new Path(outDir, "kmeans_out_" + jobID);
		FileSystem fs = FileSystem.get(configuration);
		if (fs.exists(jobOutDir)) {
			fs.delete(jobOutDir, true);
		}
		FileInputFormat.setInputPaths(job, dataDir);
		FileOutputFormat.setOutputPath(job, jobOutDir);

		Path cFile = new Path(cDir,KMeansConstants.CENTROID_FILE_PREFIX + jobID);
		System.out.println("Centroid File Path: "+ cFile.toString());
		jobConfig.set(KMeansConstants.CFILE,cFile.toString());
		jobConfig.setInt(KMeansConstants.JOB_ID, jobID);
		jobConfig.setInt(KMeansConstants.NUM_ITERATONS, numIterations);
		job.setInputFormatClass(MultiFileInputFormat.class);
		job.setJarByClass(KmeansMapCollective.class);
		job.setMapperClass(edu.iu.kmeans.allreduce.KmeansMapper.class);
		
		JobConf jobConf = (JobConf) job.getConfiguration();
		jobConf.set("mapreduce.framework.name", "map-collective");
		jobConf.setNumMapTasks(numMapTasks);
		jobConf.setInt("mapreduce.job.max.split.locations", 10000);
		job.setNumReduceTasks(0);
		jobConfig.setInt(KMeansConstants.VECTOR_SIZE,vectorSize);
		jobConfig.setInt(KMeansConstants.NUM_CENTROIDS, numCentroids);
		jobConfig.set(KMeansConstants.WORK_DIR,workDirPath.toString());
		jobConfig.setInt(KMeansConstants.NUM_MAPPERS, numMapTasks);
		return job;
	}


}
