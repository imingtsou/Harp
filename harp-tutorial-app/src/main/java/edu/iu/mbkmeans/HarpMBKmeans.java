package edu.iu.mbkmeans;

import edu.iu.fileformat.MultiFileInputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

/**
 * Created by Ethan on 4/10/17.
 */
public class HarpMBKmeans {
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        System.out.println("*********************************************");
        System.out.println("*           Harp Mini-batch Kmeans          *");
        System.out.println("*********************************************");


        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        if (args.length < 7) {
            String Usage = "Usage:: \n"
                    + "hadoop jar <jar-file-name> edu.iu.mbkmeans.HarpMBKmeans "
                    + "[input] [dimension] [numOfCentroids] [mini-batch size] [iterations] [numMapTasks] [output]\n";
            System.out.println(Usage);
            System.exit(-1);
        }

        String inputDir = args[0];
        int dimension = Integer.parseInt(args[1]);
        int K = Integer.parseInt(args[2]);
        int batchSize = Integer.parseInt(args[3]);
        int iterations = Integer.parseInt(args[4]);
        int numMapTasks = Integer.parseInt(args[5]);
        String outputDir = args[6];

        Path outputPath = new Path(outputDir);
        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(outputPath)) {
            fs.delete(outputPath, true);
        }

        conf.setInt(MBKmeansConstants.DIMENSION, dimension);
        conf.setInt(MBKmeansConstants.K, K);
        conf.setInt(MBKmeansConstants.BATCH_SIZE, batchSize);
        conf.setInt(MBKmeansConstants.NUM_ITERATONS, iterations);
        conf.setInt(MBKmeansConstants.NUM_MAP_TASKS, numMapTasks);


        Job job =  Job.getInstance(conf, "harp mini-batch kmeans");
        JobConf jobConf = (JobConf) job.getConfiguration();
        jobConf.set("mapreduce.framework.name", "map-collective");
        jobConf.setNumMapTasks(numMapTasks);
        jobConf.setNumReduceTasks(0);
        job.setJarByClass(HarpMBKmeans.class);
        job.setMapperClass(MBKmeansMapper.class);
        job.setInputFormatClass(MultiFileInputFormat.class);
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(inputDir));
        FileOutputFormat.setOutputPath(job, new Path(outputDir));
        System.exit(job.waitForCompletion(true) ? 0: 1);
    }
}
