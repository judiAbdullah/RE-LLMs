/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.cc.javaparse;

/**
 *
 * @author judi_
 */
public class BracketFix {
    public static boolean checkBrackets(String code) {
        int curlyBracesCount = 0; // To track '{' and '}'
        int parenthesesCount = 0; // To track '(' and ')'
     
        for (int i = 0; i < code.length(); i++) {
            char c = code.charAt(i);
            if (c == '{') {
                curlyBracesCount++;
            }
            else if (c == '}') {
                curlyBracesCount--;
                if (curlyBracesCount < 0) {
                    return false;
                }
            }
            else if (c == '(') {
                parenthesesCount++;
            }
            else if (c == ')') {
                parenthesesCount--;
                if (parenthesesCount < 0) {
                    return false;
                }
            }
        }
        if (curlyBracesCount > 0) {
            return false;
        } else if (parenthesesCount > 0) {
            return false;
        }
        return true;
    }

    // Function to fix unmatched brackets (add missing closing brackets)
    public static String fixBrackets(String code) {
        int curlyBracesCount = 0;
        int parenthesesCount = 0;
        StringBuilder fixedCode = new StringBuilder(code);
        for (int i = 0; i < code.length(); i++) {
            char c = code.charAt(i);
            if (c == '{') {
                curlyBracesCount++;
            }
            else if (c == '}') {
                curlyBracesCount--;
            }
            else if (c == '(') {
                parenthesesCount++;
            }
            else if (c == ')') {
                parenthesesCount--;
            }
        }
        for (int i = 0; i < parenthesesCount; i++) {
            fixedCode.append(")");
        }
        for (int i = 0; i < curlyBracesCount; i++) {
            fixedCode.append("}");
        }
        return fixedCode.toString();
    }
    
    public static void main(String[] args) {
        String code = """
                      protected RequestPredicate acceptsTextHtml() {
                      		return (serverRequest) -> {
                      			try {
                      				List<MediaType> acceptedMediaTypes = serverRequest.headers().accept();
                      				acceptedMediaTypes.remove(MediaType.ALL);
                      				MediaType.sortBySpecificityAndQuality(acceptedMediaTypes);
                      				return acceptedMediaTypes.stream()
                      						.anyMatch(MediaType.TEXT_HTML::isCompatibleWith);
                      			}
                      """;  // Example code with mismatched brackets
        
        if (checkBrackets(code)) {
            System.out.println("Brackets are balanced.");
        } else {
            System.out.println("Brackets are not balanced. Attempting to fix...");
            String fixedCode = fixBrackets(code);
            System.out.println("Fixed code: " + fixedCode);
        }
    }
}