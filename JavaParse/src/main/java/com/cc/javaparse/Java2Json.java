package com.cc.javaparse;


import com.github.javaparser.ast.Node;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.FileWriter;
import java.io.IOException;
import org.json.JSONException;

/**
 *
 * @author judi_
 */
public class Java2Json {
    
    public static JsonObject astToJson(Node node) throws JSONException{
        JsonObject jsonObject = new JsonObject();

        // Add node type
        jsonObject.addProperty("type", node.getClass().getSimpleName());

        // Add the node's text content (the code represented by this node)
        jsonObject.addProperty("value", node.toString().replaceAll("\n", " "));

        // Add the node's position in the source code
        node.getRange().ifPresent(range -> {
            JsonObject rangeObj = new JsonObject();
            rangeObj.addProperty("beginLine", range.begin.line);
            rangeObj.addProperty("beginColumn", range.begin.column);
            rangeObj.addProperty("endLine", range.end.line);
            rangeObj.addProperty("endColumn", range.end.column);
            jsonObject.add("position", rangeObj);
        });

        // Recursively process child nodes
        JsonArray children = new JsonArray();
        for (Node child : node.getChildNodes()) {
            children.add(astToJson(child));
        }

        if (children.size() > 0) {
            jsonObject.add("children", children);
        }

        return jsonObject;
        

    }
    
    public static String JsonPrettyPrinting(JsonObject jsonObject) throws JSONException{
        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        return gson.toJson(jsonObject);
    }
    
    public static void WriteToFile(String jsonObject, String path){
        // Write the JSON object to a file
        try (FileWriter file = new FileWriter(path)) {
            // Use the toString method with an indent factor for pretty printing
            file.write(jsonObject);
            System.out.println("Successfully wrote JSON object to file.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
