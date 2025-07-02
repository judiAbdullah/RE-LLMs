package com.cc.javaparse;


import com.github.javaparser.ast.Node;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringWriter;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

/**
 *
 * @author judi_
 */
public class Java2XML {
    
    public static Document ast2xml(Node node) throws ParserConfigurationException{
        // Initialize the XML document
        DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder docBuilder = docFactory.newDocumentBuilder();
        Document doc = docBuilder.newDocument();
        // Start conversion with root element
        Element rootElement = createElementForNode(doc, node);
        doc.appendChild(rootElement);

        return doc;
    }
    
    // Function to create an XML element for a given AST node
    public static Element createElementForNode(Document doc, Node node) {
        // Create the XML element for the node
        Element xmlNode = doc.createElement(node.getClass().getSimpleName());
        // Set the node's string representation as an attribute
        xmlNode.setAttribute("value", node.toString().replaceAll("\n", " "));
        // Recursively process children and add them to the current element
        for (Node child : node.getChildNodes()) {
            Element childElement = createElementForNode(doc, child);
            xmlNode.appendChild(childElement);
        }
        return xmlNode;
    }
    
    // Function to write the XML document to a file
    public static String writeXmlToString(Document doc) throws TransformerConfigurationException, TransformerException {
        TransformerFactory tf = TransformerFactory.newInstance();
        Transformer transformer = tf.newTransformer();

        // Set formatting properties (optional)
        transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "no");
        transformer.setOutputProperty(OutputKeys.METHOD, "xml");
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        transformer.setOutputProperty(OutputKeys.ENCODING, "UTF-8");

        StringWriter writer = new StringWriter();
        transformer.transform(new DOMSource(doc), new StreamResult(writer));

        return writer.getBuffer().toString();
    }
    
    public static void WriteToFile(String xmlObject, String path){
        // Write the JSON object to a file
        try (FileWriter file = new FileWriter(path)) {
            // Use the toString method with an indent factor for pretty printing
            file.write(xmlObject);
            System.out.println("Successfully wrote JSON object to file.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
