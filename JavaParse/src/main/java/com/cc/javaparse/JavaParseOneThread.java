package com.cc.javaparse;


import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.google.gson.JsonObject;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.List;
import org.w3c.dom.Document;

/**
 *
 * @author judi_
 */
public class JavaParseOneThread {
    public static void main(String[] args) {
        System.out.println("start processing ");
        long start = System.currentTimeMillis();
        
        List<String> inputList = new ArrayList<>();
        // Populate the list with some data
        for (int i = 0; i < 200000; i++) {
            String s = "public class  { public static void test" + i +"(int i, String s){ System.out.println(\"hello general\"); } }";
            inputList.add(s);
        }
        // Process the list 
        List<String> processedList = new ArrayList<>();
        for (String element : inputList) {
            processedList.add(processElement(element));  // Process each element
        }
        long end = System.currentTimeMillis();
        NumberFormat formatter = new DecimalFormat("#0.00000");
        System.out.print("Execution time is " + formatter.format((end - start) / 1000d) + " seconds");
        
    }
    
    
    
    // Function to process the elements
    public static String processElement(String element) {
        String xml = null;
        JsonObject jo = null;
        try {
            CompilationUnit cu = ParseCode(element);
            Document doc = Java2XML.ast2xml(cu);
            xml = Java2XML.writeXmlToString(doc);
//            Java2XML.WriteToFile(xml, "xml.xml");
            
            jo = Java2Json.astToJson(cu);
            String jsonstring = Java2Json.JsonPrettyPrinting(jo);
//            Java2Json.WriteToFile(jsonstring, "json.json");
        }catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
        if (xml == null){
            return null;
        }else{
            return xml;
        }
        
    }
    
    
    public static CompilationUnit ParseCode(String code){
        // Parse the code
        JavaParser parser = new JavaParser();
        CompilationUnit compilationUnit = parser.parse(code).getResult().orElseThrow();
        return compilationUnit;
    }
}
