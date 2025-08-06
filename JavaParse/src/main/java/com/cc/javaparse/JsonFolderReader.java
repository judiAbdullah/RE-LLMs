package com.cc.javaparse;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.json.JSONException;
import org.json.JSONObject;
/**
 *
 * @author judi_
 */
public class JsonFolderReader {
    public String inputfolderPath;
    public String outputfolderPath;
    public Map<String, List<JSONObject>> fileDataMap = new TreeMap<>();
    
    public JsonFolderReader(String inputPath, String outputpath){
        if (inputPath != null){
            this.inputfolderPath = inputPath;
            this.outputfolderPath = outputpath;
        }else{
            String projectRootPath = System.getProperty("user.dir");
            File projectRoot = new File(projectRootPath);
            File parentFolder = projectRoot.getParentFile();
            // this.inputfolderPath = parentFolder.getAbsolutePath() + "\\dataset\\java_dataset\\decompressedData\\";
            // this.outputfolderPath = parentFolder.getAbsolutePath() + "\\dataset\\java_dataset\\parsedData\\";
            
            // Get the parent folder path
            String parentFolderPath = parentFolder.getAbsolutePath();
            // Replace backslashes with the appropriate separator for the current platform
            String separator = File.separator;
            // Construct the paths using the separator
            this.inputfolderPath = parentFolderPath + separator + "dataset" + separator + "java_dataset" + separator + "decompressedData" + separator;
            this.outputfolderPath = parentFolderPath + separator + "dataset" + separator + "java_dataset" + separator + "parsedData" + separator;
            
            // Create a File object for the output folder
            File outputFolder = new File(this.outputfolderPath);
            // Check if the folder exists
            if (!outputFolder.exists()) {
                // Attempt to create the folder
                if (outputFolder.mkdirs()) {
                    System.out.println("Folder created successfully: " + this.outputfolderPath);
                } else {
                    System.out.println("Failed to create folder: " + this.outputfolderPath);
                }
            } else {
                System.out.println("Folder already exists: " + this.outputfolderPath);
            }
        }
        this.creatMap();
    }
    
    /**
     * to discover all files in a folder
     */
    public void creatMap(){
        File folder = new File(this.inputfolderPath);
        // Filter JSON files from the folder
        File[] jsonFiles = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".jsonl"));
        if (jsonFiles != null) {
            for (File file : jsonFiles) {
                this.fileDataMap.put(file.getName(), null);
            }
        } else {
            System.out.println("No JSON files found in the specified folder.");
        }
    }
    
    /**
     * to read json data from a specific file
     */
    public List<JSONObject> readJsonFile(String path) throws JSONException {        
        List<JSONObject> jsonObjects = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(this.inputfolderPath+path))) {
            String line;
            while ((line = reader.readLine()) != null) {
                // Parse the line into a JSONObject
                JSONObject jsonObject = new JSONObject(line);
                jsonObjects.add(jsonObject);
            }
        } catch (Exception e) {
            e.printStackTrace();  // Handle file reading exceptions
        }
        return jsonObjects;
    }
    
    
    public void writeJsonFile(List<JSONObject> parseddata, String fileName){
        try (FileWriter fileWriter = new FileWriter(this.outputfolderPath+fileName)) {
            for (JSONObject jsonObject : parseddata) {
                fileWriter.write(jsonObject.toString());
                fileWriter.write(System.lineSeparator());
            }
            System.out.println("Successfully saved JSON list to file: " + this.outputfolderPath+fileName);
        } catch (Exception e) {
            System.err.println("Error saving JSON list to file: " + e.getMessage());
        }
    }
    
}
