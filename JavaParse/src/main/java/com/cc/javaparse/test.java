/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.cc.javaparse;

import java.util.Comparator;

/**
 *
 * @author judi_
 */

public class test{

    public static void main(String[] args) {
    Comparator<Integer> comparator = (a, b) -> {
           if (a > b) return 1;
           else if (a < b) return -1;
           else return 0;
       };
    System.err.println(comparator);
    }
}