package com.cc.javaparse;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.TransformerException;

import org.w3c.dom.Document;

import static com.cc.javaparse.JavaParse.ParseCode;
import com.github.javaparser.ast.CompilationUnit;

/**
 *
 * @author judi_
 */
public class testOneNode {
    public static void main(String[] args) {
        try {

// // functions type and parameters
//            String javacode = """
// class Outer{
//    public void tf1(MyClass this,
//                    final String name, 
//                    int ... numbers1, 
//                    int a, 
//                    int b, 
//                    int[] numbers2, 
//                    Person person, Object[] objects, 
//                    List<String> list, 
//                    Function<Integer, 
//                    Integer> operation,
//                    @NonNull String info) {
//         System.out.println("tf1");
//         String sss = new String();
//         sss.toString();
//    }
// //    public static void noParams() {
// //        System.out.println("This method has no parameters.");
// //    }
// //    public <T> void tf2(T element) {
// //        System.out.println(element);
// //    }
// //    public File tf3(OuterClass OuterClass.this, String filePath) throws IOException, SQLException {
// //        // Code to read the file
// //        return new File(filePath);
// //    }
// } """;

// //    variable declaration and object creation
//            String javacode = """
// class Outer{
//    public void test() {
//        String s = new String("hello string");
//        final String CONSTANT_TEXT = "Constant";
//        String s;
//        int number = 10;
//        String text = "Hello";
//        int x = 0, y = 1, z = 2;
//        final int CONSTANT_NUMBER = 100;
//        static int sharedNumber = 5;
//        int[] numbers = new int[5];
//        String[] words = {"Hello", "World"};
//        int number = (int) Math.random() * 100;
//        String text = "Length: " + 10;
//        for (int i = 0; i < 5; i++) {
//            // i is scoped only within this block
//        }
//        double d = 10.5;
//        int i = (int) d;
//        var inferredNumber = 10;
//        var inferredText = "Hello";
//        enum Day { MONDAY, TUESDAY, WEDNESDAY }
//        Day today = Day.MONDAY;
//        Optional<String> optionalText = Optional.of("Hello");
//        volatile boolean isActive;
//        transient int temporaryData;
//        Runnable task = new Runnable() {
//            @Override
//            public void run() {
//                System.out.println("Anonymous class run method");
//            }
//        };
//        Runnable task = () -> System.out.println("Lambda run method");
//        List<String> list = new ArrayList<>();
//        Map<String, Integer> map = new HashMap<>();
//    }
// } """;

// //    assertion
//            String javacode = """
// class Outer{
//    public void test() {        
//        assert a > 0;
//        assert a > 10 : "Value is less than 10";
//        assert a > 0 : () -> "Value is: " + a;
//        assert a > 0 : {
//            System.out.println("Value is less than 0!");
//            return "Value is: " + a;
//        };
//        do{
//            int y = 1;
//            x += y;
//        }while(x < 10);
//        if (x > 0) ;
//        System.out.println("Hello, " + name);
//        for (int i=0; i < 10; i++) {
//            System.out.println(number);
//        }
//        for (int number : numbers1) {
//            System.out.println(number);
//        }
//        System.out.println(person.getName()+person.getName2());
//        for (Object obj : objects) {
//            System.out.println(obj);
//        }
//        for (String item : list) {
//            System.out.println(item);
//        }
//        int result = operation.apply(num);
//        System.out.println(result);
//        System.out.println(info);
//    }
// } """;

// String javacode = """
//     class Outer{
//     public void test(X te) {
//         int x = +10;
//         int y = -x;
//         int y = ++x;
//         int y = x++;
//         boolean a = true;
//         boolean b = !a;
//         x = -te.tes();
//     }
//     } 
// """;

// //          typeExpr
//            String javacode = """
// class Outer{
//    public void test(X te) {
//        int number = 42;
//        String str = "Hello";
//        Integer number = 10;
//        boolean isActive = true;
//        List<String> names = new ArrayList<>();
//        int[] array = new int[5]; // ArrayType
//        String[][] grid = new String[3][3]; // ArrayType with multiple levels
//        String[] words = {"apple", "banana"};
//        int[] nums = {1, 2, 3}; // Single-dimensional ArrayType
//        String[][] names = {{"John", "Doe"}, {"Jane", "Smith"}}; // Multi-dimensional ArrayType
//        List<?> unbounded = new ArrayList<>(); // WildcardType with "?"
//        List<? extends Number> numbers = new ArrayList<>(); // WildcardType with "extends"
//        List<? super Integer> integers = new ArrayList<>(); // WildcardType with "super"
//        int x = 10;
//        boolean flag = false;
//        char letter = 'A';
//        byte smallNumber = 127;
//        short shortNumber = 1000;
//        long largeNumber = 1000000000L;
//        float decimalNumber = 3.14F;
//        double preciseDecimal = 3.14159265358979;
//        Map<String, Integer> scoreMap = new HashMap<>(); // ClassOrInterfaceType with TypeArgumentList
//        enum Direction { NORTH, SOUTH, EAST, WEST }
//        Direction d = Direction.NORTH;
//        Class<Void> voidClass = void.class;
//    }
//    public void sayHello() { // VoidType
//        System.out.println("Hello!");
//    }
// } 
// class Box<T> { // TypeParameter with SimpleName
//    T value;
// }
// class BoundedBox<T extends Number> { // TypeParameter with "extends" ReferenceType
//    T value;
// }
// class Pair<K, V> { // Two TypeParameters
//    private K key;
//    private V value;
// }
// class BoundedPair<K extends Comparable<K>, V> { // TypeParameter with bounds
//    private K key;
//    private V value;
// }
//                              """;

// //           this and super
//            String javacode = """
// class Outer{
//    String name;
//    int age;
//    Outer(){}
//    Outer(String name) {
//        this(name, 0);
//        this.name = name; // Refers to the current object's 'name'  
//    }
//    Outer(String name, int age) {
//        this.name = name;
//        this.age = age;
//        this.test();
//    }
//     void test(){
//         this.name = name;
//         this.age = age;
//         this.test();
//     }
//    public Outer display() {
//        System.out.println(this);
//        return this;
//    }
//    class Inner {
//        void show() {
//            System.out.println(Outer.this); // Refers to the Outer class instance
//        }
//    }
// }
// class Child extends Outer {
//    Child(String name) {
//        super(name); // Calls parent class constructor
//        System.out.println(super.name);
//        super.display();
//    }
//    Child(){
//      super();
//    }
// }
//                              """;

// // PatternExpr
//            String javacode = """
// public class PatternExpressionDemo {
//    public void demonstratePatterns(Object input) {
//        if (input instanceof String someString) {
//            System.out.println("Matched a String: " + someString);
//        }
//        if (input instanceof Person(String name, int age)) {
//            System.out.println("Matched a Person record: Name = " + name + ", Age = " + age);
//        }
//    }
//    // Record class to use in RecordPatternExpr
//    public record Person(String name, int age) {}
// }
//                              """;

// // foreach
//            String javacode = """
// public class testclass {
//    public void test() {
//        List<String> names = List.of("Alice", "Bob", "Charlie");
//        for (String name : names) {
//            System.out.println(name);
//        }
//        Set<String> uniqueNames = Set.of("Alice", "Bob", "Charlie");
//        for (String name : uniqueNames) {
//            System.out.println(name);
//        }
//        Map<String, Integer> map = Map.of("Alice", 25, "Bob", 30, "Charlie", 35);
//        for (String key : map.keySet()) {
//            System.out.println("Key: " + key);
//        }
//        Map<String, Integer> map = Map.of("Alice", 25, "Bob", 30, "Charlie", test.getInt(35));
//        for (int value : map.values());
//        Map<String, Integer> map = Map.of("Alice", 25, "Bob", 30, "Charlie", 35);
//        for (Map.Entry<String, Integer> entry : map.entrySet()) {
//            System.out.println("Key: " + entry.getKey() + ", Value: " + entry.getValue());
//        }
//        int[][] matrix = {
//            {1, 2, 3},
//            {4, 5, 6},
//            {7, 8, 9}
//        };
//        for (int[] row : matrix) {
//            for (int num : row) {
//                System.out.print(num + " ");
//            }
//            System.out.println();
//        }
//        List<String> names = List.of("Alice", "Bob", "Charlie");
//        names.forEach(name -> System.out.println(name));
//        enum Color {
//            RED, GREEN, BLUE;
//        }
//        for (Color color : Color.values()) {
//            System.out.println(color);
//        }
//    }
// }
//                              """;

// // for
//            String javacode = """
// public class testclass {
//    public void test() {
//        for (int i = 0; i < 5; i++) {
//            System.out.println("Iteration: " + i);
//        }
//        for (;;) {
//            System.out.println("Infinite loop!");
//            break; // Avoid actual infinite loop
//        }
//        int i = 0;
//        for (; i < 3;) {
//            System.out.println("i = " + i);
//            i++;
//        }
//        for (int i = 0;;) {
//            System.out.println("i = " + i);
//            break;
//        }
//        int i = 0;
//        for (;; i++) {
//          System.out.println("i = " + i);
//          if (i >= 2) break;
//        }
//        for (int i = 0, j = 10; i < j; i++, j--) {
//            System.out.println("i = " + i + ", j = " + j);
//        }
//        for (int i = 0; i < 3; i++, System.out.println("Update executed")) {
//            System.out.println("Body executed, i = " + i);
//        }
//        outerLoop:
//        for (int i = 0; i < 3; i++) {
//            for (int j = 0; j < 3; j++) {
//                if (i == j) continue outerLoop;
//                System.out.println("i = " + i + ", j = " + j);
//            }
//        }
//        for (int i = 0; i < 3; i++) {
//            for (int j = 0; j < 3; j++) {
//                System.out.println("i = " + i + ", j = " + j);
//            }
//        }
//    }
// }
//                              """;

// //  if
//            String javacode = """
// public class testclass {
//    public void test() {
//         int x, y = 10;
//         String str = "hello";
//         if (x > 0) {
//             System.out.println("x is positive");
//         }
//         if (y > 0) {
//             System.out.println("y is positive");
//         } else {
//             System.out.println("y is non-positive");
//         }
//         if (x < y) {
//             System.out.println("x is less than y");
//         } else if (x == y) {
//             System.out.println("x is equal to y");
//         } else {
//             System.out.println("x is greater than y");
//         }
//         if (x > 0) {
//             if (y > 0) {
//                 System.out.println("Both x and y are positive");
//             }
//         }
//         if (x < 0) System.out.println("x is negative");
//         if (x > 0 && y > 0) {
//             System.out.println("Both x and y are positive");
//         }
//         if (x > 0 || y > 0) {
//             System.out.println("At least one of x or y is positive");
//         }
//         if ("Hello".equals(str)) {
//             System.out.println("The string is Hello");
//         } else if ("World".equals(str)) {
//             System.out.println("The string is World");
//         } else if ("hello".equals(str)){
//             System.out.println("The string is hello");
//         } else{
//             System.out.println("the string is somthing else");
//         }
//         if (x > y ? true : false) {
//             System.out.println("x is greater than y (evaluated via ternary)");
//         }
//         if (x > 0) {
//             int z = x + y; 
//             System.out.println("z = " + z);
//         }
//         if (isPositive(x)) {
//             System.out.println("x is positive (checked via function call)");
//         }
//         if (x > 10)
//             System.out.println("x is greater than 10");
//         if ((x > 0 && y > 0) || str.contains("test")) {
//             System.out.println("Complex condition satisfied");
//         }
//    }
// }
//                              """;

// // labeled
// String javacode = """
// public class testclass {
//    public void test() {
//        outerForLoop:
//            for (int i = 0; i < array.length; i++) {
//                for (int j = 0; j < array[i].length; j++) {
//                    if (array[i][j] == 5) {
//                        System.out.println("Breaking outer for loop at value 5");
//                        break outerForLoop; // Breaking the outer loop
//                    }
//                    System.out.println("Value: " + array[i][j]);
//                }
//            }
//    }
// }
//                              """;

// // local class
// String javacode = """
// public class testclass {
//    public void test() {
//        class LocalClass {
//            void display() {
//                System.out.println("This is a local class");
//            }
//        }
//        LocalClass local = new LocalClass();
//        local.display();
//    }
// }
//                              """;

// // return
// String javacode = """
// public class testclass {
//    public String testfunc(int value) {
//        // 1. Return a constant value
//        if (value == 1) {
//            return "One"; // Return a constant
//        }
//        // 2. Return a variable
//        String message = "Value is " + value;
//        if (value == 2) {
//            return message; // Return a variable
//        }
//        // 3. Return an expression
//        if (value == 3) {
//            return "Value: " + (value * 10); // Return a calculated expression
//        }
//        // 4. Return a method call
//        if (value == 4) {
//            return getFormattedValue(value); // Return the result of a method
//        }
//        // 5. Return from a lambda (inside anonymous class for compatibility)
//        Runnable runnable = () -> {
//            if (value == 5) {
//                return; // Return from a lambda expression
//            }
//        };
//        runnable.run();
//        // 6. Early return (no value for void methods)
//        if (value < 0) {
//            System.out.println("Negative value detected");
//            return null; // Early return to terminate execution
//        }
//        // 7. Return null
//        if (value == 6) {
//            return null; // Explicitly return null
//        }
//        // Default return
//        return "Default"; // Fallback return statement
//    }
//    private String getFormattedValue(int value) {
//        return "Formatted value: " + value;
//    }
// }
//                              """;

// // switch
// String javacode = """
// public class testclass {
//    public String testfunc(int value) {
//        String result;
//        // 1. Classic switch statement with break
//        switch (this.getv()) {
//            case this.getv():
//                System.out.println("v=1");
//                result = "One";
//                {
//                  System.out.println("inblock");
//                }
//                break;
//            case 2:
//                result = "Two";
//                break;
//            default:
//                result = "Default";
//                break;
//        }
//        // 2. Switch statement with fall-through
//        switch (value) {
//            case 3:
//            case 4:
//                result = "Three or Four";
//                break;
//            case 5:
//                result = "Five";
//                break;
//        }
//        // 3. Switch statement without default
//        switch (value) {
//            case 6:
//                result = "Six";
//                break;
//            case 7:
//                result = "Seven";
//                break;
//        }
//        // 4. Switch expression (introduced in Java 14)
//        result = switch (value) {
//            case 8 -> "Eight";
//            case 9 -> test(9);
//            default -> "Default from switch expression";
//        };
//        // 5. Switch statement with multiple case labels
//        switch (value) {
//            case 10, 11, 12:
//                result = "Ten, Eleven, or Twelve";
//                break;
//            default:
//                result = "Other";
//                break;
//        }
//        // 6. Switch expression with block body
//        result = switch (value) {
//            case 13 -> {
//                System.out.println("Processing thirteen");
//                yield "Thirteen";
//            }
//            case 14 -> {
//                System.out.println("Processing fourteen");
//                yield "Fourteen";
//            }
//            default -> {
//                System.out.println("Processing default");
//                yield "Default with block";
//            }
//        };
//        // 7. Switch with enum
//        enum Color { RED, GREEN, BLUE }
//        Color color = value % 2 == 0 ? Color.RED : Color.BLUE;
//        switch (color) {
//            case RED -> result = "Color is RED";
//            case GREEN -> result = "Color is GREEN";
//            case BLUE -> result = "Color is BLUE";
//        }
//        // Return the final result for demonstration
//        return result;
//    }
//    public int getv(){
//        return 10;
//    }
// }
//                              """;

// // synchronized 
// String javacode = """
// public class testclass {
//    private final Object lock = new Object();
//    private int counter = 0;
//    public String testfunc(int value) {
//        synchronized (lock) {
//            counter++;
//            System.out.println("Synchronized on a specific object: " + counter);
//        }
//        synchronized (this) {
//            counter++;
//            System.out.println("Synchronized on 'this': " + counter);
//        }
//        synchronized (SynchronizedStatementExamples.class) {
//            counter++;
//            System.out.println("Synchronized on class object: " + counter);
//        }
//        synchronized (lock) {
//            synchronized (this) {
//                counter++;
//                System.out.println("Nested synchronized blocks: " + counter);
//            }
//        }
//        Thread thread1 = new Thread(() -> {
//            synchronized (lock) {
//                System.out.println("Thread 1 is executing synchronized block");
//            }
//        });
//        Thread thread2 = new Thread(() -> {
//            synchronized (lock) {
//                System.out.println("Thread 2 is executing synchronized block");
//            }
//        });
//        thread1.start();
//        thread2.start();
//        if(true){
//             System.out.println("helo");
//        }
//        try {
//            thread1.join();
//            thread2.join();
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
//    }
// }
//                              """;

// // throw
// String javacode = """
// public class ThrowStatementExamples {
//    public void demonstrateThrowStatements(int value) throws Exception {
//        // 1. Throw a standard exception
//        if (value < 0) {
//            throw new IllegalArgumentException("Value cannot be negative.");
//        }
//        // 2. Throw a custom exception
//        if (value == 0) {
//            throw new CustomException("Value cannot be zero.");
//        }
//        // 3. Throw a checked exception
//        if (value > 100) {
//            throw new Exception("Value is too large.");
//        }
//        // 4. Throw a runtime exception
//        if (value == 42) {
//            throw new RuntimeException("Unlucky value: 42");
//        }
//        // 5. Throw an exception created inline
//        if (value == 99) {
//            throw new IllegalStateException("Inline exception created.");
//        }
//        // 6. Throw an exception stored in a variable
//        Exception exception = new NullPointerException("Stored exception example.");
//        if (value == 77) {
//            throw exception;
//        }
//        // 7. Throw an exception in a nested block
//        if (value > 50) {
//            try {
//                if (value == 60) {
//                    throw new ArrayIndexOutOfBoundsException("Nested throw statement.");
//                }
//            } catch (Exception e) {
//                System.out.println("Caught exception: " + e.getMessage());
//                throw e; // Rethrow the exception
//            }
//        }
//        System.out.println("Value is acceptable: " + value);
//    }
//    // Custom exception class
//    static class CustomException extends Exception {
//        public CustomException(String message) {
//            super(message);
//        }
//    }
// }
//                              """;

// // try
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String filePath) {
//        // 1. Try with multiple catch blocks
//        try {
//            int result = 10 / 0; // Causes ArithmeticException
//        } catch (ArithmeticException | NullPointerException e) {
//            System.out.println("Caught ArithmeticException: " + e.getMessage());
//        } catch (Exception e) {
//            System.out.println("Caught generic Exception: " + e.getMessage());
//        }
//        // 2. Try with finally
//        try {
//            System.out.println("Trying risky operation...");
//        } finally {
//            System.out.println("Finally block always executes.");
//        }
//        // 3. Try with resources (introduced in Java 7)
//        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
//            String line;
//            while ((line = reader.readLine()) != null) {
//                System.out.println("Read line: " + line);
//            }
//        } catch (IOException e) {
//            System.out.println("Caught IOException while reading file: " + e.getMessage());
//        }
//        // 4. Try with resources and multiple resources
//        try (
//            FileReader fileReader = new FileReader(filePath);
//            BufferedReader bufferedReader = new BufferedReader(fileReader)
//        ) {
//            String content;
//            while ((content = bufferedReader.readLine()) != null) {
//                System.out.println("Content: " + content);
//            }
//        } catch (IOException e) {
//            System.out.println("Error with multiple resources: " + e.getMessage());
//        }
//        // 5. Nested try-catch
//        try {
//            try {
//                throw new NullPointerException("Nested exception");
//            } catch (NullPointerException e) {
//                System.out.println("Caught nested NullPointerException: " + e.getMessage());
//                throw e; // Rethrow exception
//            }
//        } catch (Exception e) {
//            System.out.println("Caught rethrown exception: " + e.getMessage());
//        }
//        // 6. Empty try-catch-finally block
//        try {
//        }catch (Exception e) {
//            // No action on exception
//        } finally {
//            System.out.println("Finally executed even in empty block.");
//        }
//    }
// }
//                """;

// // while
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String filePath) {
//        int counter = 0;
//        // 1. Simple while loop
//        while (counter < 5) {
//            System.out.println("Counter: " + counter);
//            counter++;
//        }
//        // 2. While loop with a break statement
//        counter = 0;
//        while (true) {
//            System.out.println("Counter in infinite loop with break: " + counter);
//            if (counter == 2) {
//                break; // Exit the loop when counter reaches 2
//            }
//            counter++;
//        }
//        // 3. While loop with a continue statement
//        counter = 0;
//        while (counter < 5) {
//            counter++;
//            if (counter % 2 == 0) {
//                continue; // Skip even numbers
//            }
//            System.out.println("Odd Counter: " + counter);
//        }
//        // 4. Nested while loops
//        counter = 0;
//        int innerCounter;
//        while (counter < 3) {
//            innerCounter = 0;
//            while (innerCounter < 2) {
//                System.out.println("Outer Counter: " + counter + ", Inner Counter: " + innerCounter);
//                innerCounter++;
//            }
//            counter++;
//        }
//        // 5. While loop with an empty body
//        counter = 0;
//        while (counter < 5)
//            counter++; // Incrementing counter without a block
//        System.out.println("Counter after empty body loop: " + counter);
//        // 6. While loop with a complex condition
//        int a = 10, b = 20;
//        while (a < b && a > 0) {
//            System.out.println("a: " + a + ", b: " + b);
//            a += 2;
//        }
//        // 7. Do-While loop example
//        counter = 0;
//        do {
//            System.out.println("Counter in do-while: " + counter);
//            counter++;
//        } while (counter < 3);
//    }
// }
//                """;

// // yield
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String input) {
//        String result = switch (input) {
//            case "A" -> {
//                System.out.println("Case A matched");
//                yield "Result A";
//            }
//            case "B" -> {
//                System.out.println("Case B matched");
//                int intermediateValue = 42
//                yield "Result B: " + intermediateValue;
//            }
//            case "C" -> "Result C"; 
//            default -> {
//                System.out.println("Default case");
//                yield "Default Result";
//            }
//        };
//        return result;
//    }
// }
//                """;

// // array access
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String input) {
//        int[] numbers = {10, 20, 30, 40, 50};
//        System.out.println("First element: " + numbers[0]);
//        System.out.println("Third element: " + numbers[2]);
//        numbers[1] = 25;
//        System.out.println("Modified second element: " + numbers[1]);
//        int[][] matrix = {
//            {1, 2, 3},
//            {4, 5, 6},
//            {7, 8, 9}
//        };
//        System.out.println("Element at (1,2): " + matrix[1][2]);
//        matrix[0][0] = 42;
//        System.out.println("Modified element at (0,0): " + matrix[0][0]);
//        int row = 2, col = 1;
//        System.out.println("Element at (row=2, col=1): " + matrix[row][col]);
//        int sum = numbers[0] + numbers[4];
//        System.out.println("Sum of first and last elements: " + sum);
//        String[] words = {"hello", "world", "java"};
//        System.out.println("First word: " + words[0]);
//        System.out.println("Length of second word: " + words[1].length());
//        for (int i = 0; i < numbers.length; i++) {
//            System.out.println("Element at index " + i + ": " + numbers[i]);
//        }
//    }
// }
//                """;

// // array creation
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String input) {
//        int[] singleDimArray = new int[5];
//        singleDimArray[0] = 10;
//        singleDimArray[1] = 20;
//        int[] initializedArray = {1, 2, 3, 4, 5};
//        System.out.println("Initialized array first element: " + initializedArray[0]);
//        int[][] multiDimArray = new int[3][4];
//        multiDimArray[0][0] = 42;
//        int[][] partiallySizedArray = new int[3][];
//        partiallySizedArray[0] = new int[2];
//        String[] stringArray = new String[3];
//        stringArray[0] = "Hello";
//        stringArray[1] = "World";
//        String[][] stringMatrix = new String[2][2];
//        stringMatrix[0][0] = "Java";
//        stringMatrix[1][1] = "Programming";
//        int sum = new int[]{1, 2, 3}[0] + new int[]{4, 5, 6}[1];
//        System.out.println("Sum of array elements: " + sum);
//        System.out.println("Anonymous array access: " + new int[]{10, 20, 30}[2]);
//        int[][] dynamicArray = new int[2][];
//        for (int i = 0; i < dynamicArray.length; i++) {
//            dynamicArray[i] = new int[i + 1];
//        }
//    }
// }
//                """;

// // assign
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String input) {
//        int x = 10;
//        x += 5;
//        x -= 3;
//        x %= 3;
//        x &= y;
//        x |= y;
//        x ^= y;
//        x <<= 2;
//        x >>= 1;
//        x >>>= 1;
//        Dummy dummy = new Dummy();
//        dummy.value = 50;
//        x = y = z = 100;
//        x = (y > z) ? y : z;           
//    }
//    class Dummy {
//            int value;
//        }
// }
//                """;

// // binary expression 
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String input) {
//         int x = 10, y = 20, z = 30;
//         int sum = x + y;
//         int difference = y - x;
//         int product = x * y;
//         int quotient = y / x;
//         int remainder = y % x;
//         System.out.println("Sum: " + sum);
//         boolean isEqual = x == y; // Equality
//         boolean isNotEqual = x != y; // Inequality
//         boolean isGreater = y > x; // Greater than
//         boolean isLess = x < y; // Less than
//         boolean isGreaterOrEqual = y >= x; // Greater than or equal to
//         boolean isLessOrEqual = x <= y; // Less than or equal to
//         System.out.println("x == y: " + isEqual);
//         boolean andResult = (x < y) && (y < z); // Logical AND
//         boolean orResult = (x > y) || (y < z); // Logical OR
//         int andBitwise = x & y; // Bitwise AND
//         int orBitwise = x | y; // Bitwise OR
//         int xorBitwise = x ^ y; // Bitwise XOR
//         int leftShift = x << 2; // Left shift
//         int rightShift = x >> 1; // Right shift
//         int unsignedRightShift = x >>> 1; // Unsigned right shift
//         boolean conditionalAnd = (x < y) && (y > z); // Conditional AND
//         boolean conditionalOr = (x > y) || (y < z); // Conditional OR
//    }
// }
//                """;

// //  cast
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String input) {
//        double pi = 3.14159;
//        int intPi = (int) pi;
//        Integer integerValue = Integer.valueOf(100);
//        int primitiveInt = (int) integerValue;
//        Object obj = "Hello, World!";
//        String str = (String) obj;      
//        Object arrayObj = new int[]{1, 2, 3};
//        int[] intArray = (int[]) arrayObj;
//    }
// }
//                """;

// //  class expr
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String input) {
//        Class<String> stringClass = String.class;
//        System.out.println("Class for String: " + stringClass.getName());
//        Class<int[]> intArrayClass = int[].class;
//        System.out.println("Class for int[]: " + intArrayClass.getName());
//        Class<Integer> intClass = int.class;
//        System.out.println("Class for int: " + intClass.getName());
//        Class<Void> voidClass = void.class;
//        System.out.println("Class for void: " + voidClass.getName());
//        Class<ClassExpressionExamples> exampleClass = ClassExpressionExamples.class;
//        System.out.println("Class for ClassExpressionExamples: " + exampleClass.getName());
//        Class<Thread.State> enumClass = Thread.State.class;
//        System.out.println("Class for Thread.State enum: " + enumClass.getName());
//        Class<NestedClass> nestedClass = NestedClass.class;
//        System.out.println("Class for NestedClass: " + nestedClass.getName());
//        Class<java.util.List> listClass = java.util.List.class;
//        System.out.println("Class for List: " + listClass.getName());
//    }
// }
//                """;

// //  ConditionalExpr
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String input) {
//        boolean isWeekend = true;
//        String activity = isWeekend ? getWeekendActivity() : getWeekdayActivity();
//        System.out.println("Activity: " + activity);
//    }
// }
//                """;

// //  enclose expr
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String input) {
//        int result = (5 + 3) * 2;
//        boolean isTrue = !(false || (5 > 3));
//        int a = 10, b = 20;
//        int max = ((a > b) ? a : b);
//    }
// }
//                """;

// //  field access expr
// String javacode = """
// public class TryStatementExamples {
//    public static void demonstrateTryStatements(String input) {
//        Point point = new Point(10, 20);
//        point.test().get().print();
//        point.toString();
//        int xValue = point.x;
//        double pi = Math.PI;
//        int yValue = new Point(30, 40).y;
//        Rectangle rectangle = new Rectangle(new Point(5, 10), new Point(15, 20));
//        int rectX1 = rectangle.topLeft.x;
//        Circle circle = new Circle();
//        int centerX = circle.center.x;
//        Point[] points = {new Point(1, 2), new Point(3, 4)};
//        int arrayX = points[1].x;
//        int chainedAccess = circle.center.y + rectangle.bottomRight.x;
//        SubPoint subPoint = new SubPoint(50, 60, 70);
//        int inheritedX = subPoint.x;
//        String substringClassName = "hello".substring(0, 2).getClass().getSimpleName();
//    }
// }
//                """;

// //  LambdaExpr
// String javacode = """
// public class TryStatementExamples {
//    public void demonstrateTryStatements(String input) {
//        Runnable noParam = () -> System.out.println("Lambda with no parameters");
//        noParam.run();
//        Consumer<String> singleParam = (message) -> System.out.println("Message: " + message);
//        singleParam.accept("Hello, Lambda!");
//        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
//        BiConsumer<String, Integer> blockLambda = (name, age) -> {
//            System.out.println("Name: " + name);
//            System.out.println("Age: " + age);
//        };
//        blockLambda.accept("Alice", 30);
//        Consumer<Integer> inferredType = x -> System.out.println("Square: " + (x * x));
//        inferredType.accept(5);
//        BiFunction<Double, Double, Double> multiply = (Double x, Double y) -> x * y;
//        Comparator<Integer> comparator = (a, b) -> {
//            if (a > b) return 1;
//            else if (a < b) return -1;
//            else return 0;
//        };
//        Supplier<String> supplyString = () -> {
//            String greeting = "Hello";
//            String name = "World";
//            return greeting + ", " + name + "!";
//        };
//        Function<String, Integer> stringLength = (str) -> str.length();
//        Function<Integer, Function<Integer, Integer>> createAdder = (a) -> (b) -> a + b;
//        Function<Integer, Integer> add5 = createAdder.apply(5);
//    }
// }
//                """;

// //  Method call expr
// String javacode = """
// public class TryStatementExamples {
//     public void demonstrateTryStatements(String input) {
//         printHello();
//         printMessage("Hello, MethodCallExpr!");
//         int sum = addNumbers(5, 10);
//         System.out.println("Sum: " + sum);
//         String message = "Hello, Java!";
//         int length = message.length();
//         System.out.println("Length of the message: " + length);
//         String upperCaseMessage = message.toLowerCase().replace("java", "MethodCallExpr").toUpperCase();
//         System.out.println("Transformed message: " + upperCaseMessage);
//         double randomValue = Math.random();
//         System.out.println("Random value: " + randomValue);
//         printMessage("The length of 'Java' is: " + "Java".length());
//         List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
//         numbers.forEach(num -> System.out.println("Number: " + num));
//         String conditionalMessage = getMessage(true ? "Yes" : "No");
//         System.out.println(conditionalMessage);
//         processString(getMessage("Chained Method Call"));
//         this.printMessage("Method call on 'this' object");
//         superMethodCallExample();
//         printVarargs(1, 2, 3, 4, 5);
//     }
//     private void printHello() {
//         System.out.println("Hello, World!");
//     }
//     private void printMessage(String message) {
//         System.out.println(message);
//     }
//     private int addNumbers(int a, int b) {
//         return a + b;
//     }
//     private String getMessage(String input) {
//         return "Message: " + input;
//     }
//     private void processString(String str) {
//         System.out.println("Processed string: " + str);
//     }
//     private void printVarargs(int... numbers) {
//         System.out.println("Varargs: " + Arrays.toString(numbers));
//     }
//     private void superMethodCallExample() {
//         ParentClass parent = new ParentClass();
//         parent.showMessage();
//     }
// }
//                 """;

// //  MethodReferenceExpr
// String javacode = """
// public class TryStatementExamples {
//     public void demonstrateTryStatements(String input) {
//         Function<String, Integer> stringToInteger = Integer::parseInt;
//         System.out.println("Parsed Integer: " + stringToInteger.apply("123"));
//         String message = "Hello, MethodReference!";
//         Supplier<Integer> getLength = message::length;
//         System.out.println("Message length: " + getLength.get());
//         Consumer<String> printMessage = System.out::println;
//         printMessage.accept("Instance Method Reference on a Parameter");
//         Supplier<List<String>> createList = ArrayList::new;
//         List<String> myList = createList.get();
//         myList.add("Constructor Reference Example");
//         System.out.println(myList);
//         Function<String, Boolean> startsWithH = message::startsWith;
//         System.out.println("Starts with 'H': " + startsWithH.apply("H"));
//         BiPredicate<String, String> stringComparator = String::equalsIgnoreCase;
//         System.out.println("Strings equal (ignoring case): " + stringComparator.test("Hello", "hello"));
//         Function<Integer, int[]> createArray = int[]::new;
//         int[] myArray = createArray.apply(5);
//         System.out.println("Array length: " + myArray.length);
//         List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
//         names.stream().map(String::toUpperCase).forEach(System.out::println);
//         MethodReferenceExpressionExamples example = new MethodReferenceExpressionExamples();
//         Runnable printExample = example::customInstanceMethod;
//         printExample.run();
//     }
//     public void customInstanceMethod() {
//         System.out.println("Custom instance method called");
//     }
// }
//                 """;

// //  MethodReferenceExpr
// String javacode = """
// public class TryStatementExamples {
//     public void demonstrateTryStatements(String input) {
//         String emptyString = new String();
//         String greeting = new String("Hello, Object Creation!");
//         int[] numbers = new int[5];
//         Person person = new Person("Alice", 25);
//         List<String> names = new ArrayList<>(List.of("Alice", "Bob", "Charlie"));
//         Runnable runnable = new Runnable() {
//             @Override
//             public void run() {
//                 System.out.println("Anonymous Runnable running!");
//             }
//         };
//     }
// }
//                 """;


// //  Instanceof
// String javacode = """
// public class TryStatementExamples {
//     public void demonstrateTryStatements(String input) {
//         // pattern instanceof
//         if (input instanceof String someString) {
//             System.out.println("Matched a String: " + someString);
//         }
//         if (input instanceof Person(String name, int age)) {
//             System.out.println("Matched a Person record: Name = " + name + ", Age = " + age);
//         }
//     }
// }
//                 """;


String javacode = """
class Test{
    public static void my_method() {
        if (x.check(x)) {
            System.out.println("x");
            if (2){
                System.out.println("x");
            }else{
                return;
            }
        } else if (obj.func()) {
            return;
        }

        // switch (value) {
        //     case 1:
        //         System.out.println("Matched 1");
        //         break;
        //     case 2: 
        //         System.out.println("Matched 2");
        //         break;
        //     default:
        //         if (value == 3) {
        //             System.out.println("Matched 3");
        //         }
        // }
        
        // TestClass obj = new Class1();
        // for (int x : new int[]{1, 2, 3, 4}) {
        //     obj.method(x);
        // }

        // try {
        //     System.out.println("try body");
        // } catch (Exception e) {
        //     System.out.println(e.getMessage());
        // } finally {
        //     System.out.println("finally");
        // }
    }
}

""";
            
            CompilationUnit cu = ParseCode(javacode);

            if(cu.getChildNodes().isEmpty() && cu.toString().contains("???"))
                System.err.println("compile error");
            Document doc = Java2XML.ast2xml(cu);
            String xml = Java2XML.writeXmlToString(doc);
            Java2XML.WriteToFile(xml, "xml.xml");
            
                

            
        }catch (ParserConfigurationException | TransformerException e) {
            e.printStackTrace(System.out);
        }
    }
    
    
    public static String formatJavaCode(String code) {
        // Replace escaped new lines with actual new lines
        code = code.replace("\\n", "\n");

        // Replace tab characters with spaces
        code = code.replace("\t", "    "); // Using 4 spaces for a tab
        // Normalize spaces
        code = code.replaceAll(" +", " ");

        // Format indentation
        StringBuilder formattedCode = new StringBuilder();
        int indentLevel = 0;
        boolean inString = false;

        for (int i = 0; i < code.length(); i++) {
            char currentChar = code.charAt(i);

            // Toggle inString when encountering quotes
            if (currentChar == '\"') {
                inString = !inString;
            }

            // Adjust indentation for braces
            if (!inString) {
                switch (currentChar) {
                    case '{' -> {
                        formattedCode.append(" {\n");
                        indentLevel++;
                        appendIndentation(formattedCode, indentLevel);
                        continue; // Skip appending the '{' again
                    }
                    case '}' -> {
                        formattedCode.append("\n");
                        indentLevel--;
                        appendIndentation(formattedCode, indentLevel);
                        formattedCode.append("}");
                        continue; // Skip appending the '}' again
                    }
                    case ';' -> {
                        formattedCode.append(";\n");
                        appendIndentation(formattedCode, indentLevel);
                        continue; // Skip appending the ';' again
                    }
                    default -> {
                    }
                }
            }

            formattedCode.append(currentChar);
        }

        return formattedCode.toString().trim();
    }
    
    private static void appendIndentation(StringBuilder sb, int level) {
        for (int i = 0; i < level; i++) {
            sb.append("    "); // 4 spaces for indentation
        }
    }

}

