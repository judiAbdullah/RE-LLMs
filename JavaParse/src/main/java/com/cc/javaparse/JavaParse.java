package com.cc.javaparse;


import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.google.gson.JsonObject;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.json.JSONObject;
import org.w3c.dom.Document;

/**
 *
 * @author judi_
 */
public class JavaParse {
    public static void main(String[] args) {
        System.out.println("start processing ");
        long start = System.currentTimeMillis();
        JsonFolderReader jfd = new JsonFolderReader(null, null);
        try {
            for(Map.Entry<String, List<JSONObject>> entry  : jfd.fileDataMap.entrySet()){
                List<JSONObject> inputdata = jfd.readJsonFile(entry.getKey());
                List<JSONObject> parseddata = processInParallel(inputdata, 5);
                jfd.writeJsonFile(parseddata, entry.getKey());
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        long end = System.currentTimeMillis();
        NumberFormat formatter = new DecimalFormat("#0.00000");
        System.out.print("Execution time is " + formatter.format((end - start) / 1000d) + " seconds");

    }
    
    // Method to split the input list, process using threads, and recombine results
    public static List<JSONObject> processInParallel(List<JSONObject> inputList, int numThreads) {
        int size = inputList.size();
        int chunkSize = size / numThreads;

        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        List<Future<List<JSONObject>>> futures = new ArrayList<>();

        // Divide list into chunks and submit tasks for each chunk
        for (int i = 0; i < numThreads; i++) {
            int threadNumber = i + 1;
            int start = i * chunkSize;
            int end = (i == numThreads - 1) ? size : start + chunkSize;

            // Create a sublist for each thread to process
            List<JSONObject> subList = inputList.subList(start, end);

            // Submit a task to process this sublist
            Future<List<JSONObject>> future = executorService.submit(new Callable<List<JSONObject>>() {
                @Override
                public List<JSONObject> call() {
                    System.out.println("start sublist process" + threadNumber);
                    List<JSONObject> result = new ArrayList<>();
                    for (JSONObject element : subList) {
                        JSONObject parsedobject = processElement(element);
                        if (parsedobject != null)
                            result.add(parsedobject);  // Process each element
                    }
                    System.out.println("end of sublist process" + threadNumber);
                    return result;
                }
            });

            futures.add(future);
        }

        // Gather the results
        List<JSONObject> finalResult = new ArrayList<>();
        for (Future<List<JSONObject>> future : futures) {
            try {
                finalResult.addAll(future.get());  // Rejoin processed data
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
        executorService.shutdown();  // Shut down the executor service
        return finalResult;
    }
    
    // Function to process the elements
    public static JSONObject processElement(JSONObject element) {
        String xml = null;
        JsonObject jo = null;
        CompilationUnit cu = null;
        try {
            
            String code = "class test{ \n" + element.getString("code")+" \n}";
            cu = ParseCode(code);
            if (cu != null){
                if(cu.getChildNodes().isEmpty() && cu.toString().contains("???")){
                    boolean bracketerror = BracketFix.checkBrackets(element.getString("code"));
                    if(bracketerror){
                        String fixedcode = BracketFix.fixBrackets(element.getString("code"));
                        code = "class test{ \n" + fixedcode+" \n}";
                        cu = ParseCode(code);
                    }
                }
            }
            
            Document doc = Java2XML.ast2xml(cu);
            xml = Java2XML.writeXmlToString(doc);
            element.put("xml", xml);
            
        }catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
        if (xml == null){
            
            return null;
        }else{
            return element;
        }
        
    }
    
   


    public static CompilationUnit ParseCode(String code){
        // Parse the code
        JavaParser parser = new JavaParser();
        CompilationUnit compilationUnit = parser.parse(code).getResult().orElseThrow();
        return compilationUnit;
    }
    
    


}
