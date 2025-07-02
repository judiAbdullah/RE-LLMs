package com.cc.javaparse;

import com.github.javaparser.ast.Node;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import org.reflections.Reflections;

import java.util.Set;
/**
 *
 * @author judi_
 */
public class temptest {
    public static void main(String[] args) {
        // Use Reflections to scan for all classes extending Node
        Reflections reflections = new Reflections("com.github.javaparser.ast");

        // Get all subtypes of Node
        Set<Class<? extends Node>> allNodeTypes = reflections.getSubTypesOf(Node.class);
//        // Convert Set to List for sorting
//        List<Class<? extends Node>> sortedNodeTypes = new ArrayList<>(allNodeTypes);
//        System.out.println(allNodeTypes.size());
//        // Sort the list by class name
//        sortedNodeTypes.sort(Comparator.comparing(Class::getSimpleName));

        // Print all node types
        allNodeTypes.forEach(nodeType -> System.out.println(nodeType.getSimpleName()));
    }
}
