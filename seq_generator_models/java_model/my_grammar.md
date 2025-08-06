PackageDeclaration
ImportDeclaration
ModuleDeclaration

ConstructorDeclaration
MethodDeclaration
ConstructorDeclaration ::= { Modifier } SimpleName "(" { Parameter } ")" { Throws } ConstructorBody

ConstructorBody ::= BlockStmt

MethodDeclaration ::= { Modifier } TypeOrVoid SimpleName "(" { Parameter } ")" { Throws } MethodBody

Modifier ::= Modifier
TypeOrVoid ::= Type | VoidType
SimpleName ::= SimpleName
Parameter ::= ReceiverParameter | Parameter
ReceiverParameter ::= [ Modifier ] Type [ Identifier ]
Parameter ::= [ Modifier ] Type Identifier [ "..." ]
Throws ::= "throws" { ReferenceType }
MethodBody ::= BlockStmt | ";"

BlockStmt ::= "{" { Statement } "}"

Statement ::= AssertStmt
            | BlockStmt
            | BreakStmt
            | ContinueStmt
            | DoStmt
            | EmptyStmt
            | ExplicitConstructorInvocationStmt
            | ExpressionStmt
            | ForEachStmt
            | ForStmt
            | IfStmt
            | LabeledStmt
            | LocalClassDeclarationStmt
            | LocalRecordDeclarationStmt
            | ReturnStmt
            | SwitchStmt
            | SynchronizedStmt
            | ThrowStmt
            | TryStmt
            | UnparsableStmt
            | WhileStmt
            | YieldStmt

Expression ::= ArrayAccessExpr
             | ArrayCreationExpr
             | ArrayInitializerExpr
             | AssignExpr
             | BinaryExpr
             | CastExpr
             | ClassExpr
             | ConditionalExpr
             | EnclosedExpr
             | FieldAccessExpr
             | InstanceOfExpr
             | LambdaExpr
             | LiteralExpr
             | MethodCallExpr
             | MethodReferenceExpr
             | NameExpr
             | ObjectCreationExpr
             | PatternExpr
             | SuperExpr
             | ThisExpr
             | TypeExpr
             | UnaryExpr
             | VariableDeclarationExpr

LiteralExpr ::= BooleanLiteralExpr
              | CharLiteralExpr
              | DoubleLiteralExpr
              | IntegerLiteralExpr
              | LongLiteralExpr
              | NullLiteralExpr
              | StringLiteralExpr
              | TextBlockLiteralExpr


AssertStmt ::= "assert" Expression [ ":" Expression ] ";"
BreakStmt ::= "break" [ SimpleName ] ";"
ContinueStmt ::= "continue" [ SimpleName ] ";"
DoStmt ::= "do" Statement "while" "(" Expression ")" ";"
EmptyStmt ::= ";"
<!-- need check because the parser does not take input like Outer.super() -->
ExplicitConstructorInvocationStmt ::= [ Expression "." ] "super" "(" [ { Expression [ "," ] } ] ")" ";"
                                     | [ Expression "." ] "this" "(" [ { Expression [ "," ] } ] ")" ";"
ExpressionStmt ::= Expression ";"
ForEachStmt ::= "for" "(" VariableDeclarator ":" Expression ")" Statement
ForStmt ::= "for" "(" [ ForInit ] ";" [ Expression ] ";" [ ForUpdate ] ")" Statement
ForInit ::= VariableDeclarationExpr | Expression
ForUpdate ::= Expression
IfStmt ::= "if" "(" Expression ")" Statement [ "else" Statement ]
<!-- check with prof to specify if we need to add some label creation in seq dig -->
LabeledStmt ::= SimpleName ":" Statement
LocalClassDeclarationStmt ::= ClassOrInterfaceDeclaration
LocalRecordDeclarationStmt ::= RecordDeclaration
ReturnStmt ::= "return" [ Expression ] ";"
SwitchStmt ::= "switch" "(" Expression ")" "{" { SwitchEntry } "}"
SwitchEntry ::= CaseOrDefault ":" { Statement }
CaseOrDefault ::= "case" Expression | "default"
SynchronizedStmt ::= "synchronized" "(" Expression ")" BlockStmt
ThrowStmt ::= "throw" Expression ";"
TryStmt ::= "try" BlockStmt { CatchClause } [ "finally" BlockStmt ]
CatchClause ::= "catch" "(" Parameter ")" BlockStmt
UnparsableStmt ::= <invalid input>
WhileStmt ::= "while" "(" Expression ")" Statement
YieldStmt ::= "yield" Expression ";"









ArrayAccessExpr ::= Expression "[" Expression "]"
ArrayCreationExpr ::= "new" Type { ArrayCreationLevel } [ ArrayInitializerExpr ]
ArrayCreationLevel ::= "[" [ Expression ] "]"
ArrayInitializerExpr ::= "{" [ Expression { "," Expression } ] "}"
AssignExpr ::= Expression AssignOperator Expression
<!-- does not appear in ast -->
AssignOperator ::= "=" | "+=" | "-=" | "*=" | "/=" | "&=" | "|=" | "^=" | "%=" | "<<=" | ">>=" | ">>>="
BinaryExpr ::= Expression BinaryOperator Expression
BinaryOperator ::= "||" | "&&" | "|" | "^" | "&" | "==" | "!=" | "<" | ">" | "<=" | ">=" | "<<" | ">>" | ">>>" | "+" | "-" | "*" | "/" | "%"
CastExpr ::= "(" Type ")" Expression
<!-- return value from caller stmt or expr -->
ClassExpr ::= Type "." "class"

ConditionalExpr ::= Expression "?" Expression ":" Expression

EnclosedExpr ::= "(" Expression ")"
FieldAccessExpr ::= Expression "." SimpleName
LambdaExpr ::= "(" [ { Parameter { "," Parameter } } ] ")" "->" ( Expression | BlockStmt )

MethodCallExpr ::= [ Expression "." ] SimpleName "(" [ { Expression { "," Expression } } ] ")"

MethodReferenceExpr ::= Expression "::" SimpleName
NameExpr ::= SimpleName
ObjectCreationExpr ::= "new" Type "(" [ { Expression { "," Expression } } ] ")" [ ClassBody ]
ClassBody ::= "{" { BodyDeclaration } "}"



InstanceOfExpr ::= Expression "instanceof" Type [ PatternExpr ]
PatternExpr ::= TypePatternExpr
              | RecordPatternExpr

TypePatternExpr ::= Type SimpleName
RecordPatternExpr ::= Type "(" { VariableDeclarator { "," VariableDeclarator } } ")"

<!-- this hold just the value -->
SuperExpr ::= [ Expression "." ] "super"
ThisExpr ::= [ Expression "." ] "this"

TypeExpr ::= Type
<!-- for now we just get the value -->
Type ::= ReferenceType | PrimitiveType | VoidType
ReferenceType ::= ClassOrInterfaceType | ArrayType | TypeParameter | WildcardType
PrimitiveType ::= "int" | "boolean" | "char" | "byte" | "short" | "long" | "float" | "double"
VoidType ::= "void"
ClassOrInterfaceType ::= SimpleName [ TypeArgumentList ]
ArrayType ::= Type [ ArrayCreationLevel ]
TypeParameter ::= SimpleName [ "extends" ReferenceType { "&" ReferenceType } ]
WildcardType ::= "?" [ "extends" ReferenceType ] | "?" [ "super" ReferenceType ]
TypeArgumentList ::= "<" TypeArgument { "," TypeArgument } ">"
TypeArgument ::= Type | WildcardType


UnaryExpr ::= UnaryOperator Expression
UnaryOperator ::= "+" | "-" | "++" | "--" | "~" | "!"

<!-- javaparser change this into 
variabledeclarationexpr ::= variabledeclarator {"," variable declarator}
variabledeclarator ::= type simplename ["=" expression] -->
VariableDeclarationExpr ::= { Modifier } Type VariableDeclarator { "," VariableDeclarator }
VariableDeclarator ::= SimpleName [ "=" Expression ]

SwitchExpr ::= "switch" "(" Expression ")" "{" { SwitchEntry } "}"
SwitchEntry ::= CaseOrDefault ":" { Expression }
CaseOrDefault ::= "case" Expression | "default"












<!-- 

BooleanLiteralExpr ::= "true" | "false"
CharLiteralExpr ::= "'" Character "'"
DoubleLiteralExpr ::= DigitSequence "." DigitSequence [ Exponent ] [ FloatTypeSuffix ]
                    | "." DigitSequence [ Exponent ] [ FloatTypeSuffix ]
                    | DigitSequence Exponent [ FloatTypeSuffix ]
                    | DigitSequence FloatTypeSuffix

Exponent ::= "e" [ "+" | "-" ] DigitSequence

FloatTypeSuffix ::= "d" | "D"
IntegerLiteralExpr ::= DecimalIntegerLiteral
                     | HexIntegerLiteral
                     | OctalIntegerLiteral
                     | BinaryIntegerLiteral

DecimalIntegerLiteral ::= DigitSequence [ IntegerTypeSuffix ]
HexIntegerLiteral ::= "0x" HexDigitSequence [ IntegerTypeSuffix ]
OctalIntegerLiteral ::= "0" OctalDigitSequence [ IntegerTypeSuffix ]
BinaryIntegerLiteral ::= "0b" BinaryDigitSequence [ IntegerTypeSuffix ]

DigitSequence ::= Digit { Digit }
HexDigitSequence ::= HexDigit { HexDigit }
OctalDigitSequence ::= OctalDigit { OctalDigit }
BinaryDigitSequence ::= BinaryDigit { BinaryDigit }

IntegerTypeSuffix ::= "l" | "L"
LongLiteralExpr ::= IntegerLiteralExpr "l" | IntegerLiteralExpr "L"
NullLiteralExpr ::= "null"
StringLiteralExpr ::= "\"" { Character } "\""
TextBlockLiteralExpr ::= "\"\"\"" { TextBlockContent } "\"\"\""
TextBlockContent ::= { Character }

SimpleName ::= Identifier
Name ::= SimpleName | Name "." SimpleName
Identifier ::= Letter { Letter | Digit } 
Letter ::= "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
           | "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z"
           | "_" | "$"
Digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

BodyDeclaration ::= ConstructorDeclaration
                 | MethodDeclaration
                 | FieldDeclaration
                 | EnumConstantDeclaration
                 | AnnotationMemberDeclaration
                 | ClassOrInterfaceDeclaration
                 | RecordDeclaration
                 | LocalClassDeclarationStmt
                 | LocalRecordDeclarationStmt
ModuleDirective ::= ModuleRequiresDirective
                  | ModuleExportsDirective
                  | ModuleOpensDirective
                  | ModuleUsesDirective
                  | ModuleProvidesDirective
CatchClause ::= "catch" "(" Parameter ")" BlockStmt
Modifier ::= "public" | "protected" | "private" | "static" | "final" | "abstract" | "synchronized" | "volatile" | "transient" | "native" | "strictfp" | Annotation
MemberValuePair ::= SimpleName "=" Expression
AnnotationExpr ::= MarkerAnnotationExpr | SingleMemberAnnotationExpr | NormalAnnotationExpr

MarkerAnnotationExpr ::= "@" SimpleName
SingleMemberAnnotationExpr ::= "@" SimpleName "(" Expression ")"
NormalAnnotationExpr ::= "@" SimpleName "(" [ { MemberValuePair { "," } } ] ")"
LiteralStringValueExpr ::= "\"" { Character } "\""
CompilationUnit ::= { PackageDeclaration } { ImportDeclaration } { TypeDeclaration } [ EOF ]
Comment ::= JavadocComment | LineComment | BlockComment
InitializerDeclaration ::= BlockStmt
CompactConstructorDeclaration ::= { Modifier } SimpleName "(" [ { Parameter { "," Parameter } } ] ")" { Throws } "{" { Statement } "}"
FieldDeclaration ::= { Modifier } Type VariableDeclarator { "," VariableDeclarator } ";"
EnumConstantDeclaration ::= SimpleName [ "(" [ { Expression { "," Expression } } ] ")" ] [ ClassBody ]
TypeDeclaration ::= ClassOrInterfaceDeclaration | EnumDeclaration | AnnotationDeclaration | RecordDeclaration
CallableDeclaration ::= MethodDeclaration | ConstructorDeclaration | AnnotationMemberDeclaration
AnnotationMemberDeclaration ::= { Modifier } Type SimpleName "(" [ { Parameter { "," Parameter } } ] ")" [ DefaultValue ] ";"
DefaultValue ::= "default" ElementValue
ModuleProvidesDirective ::= "provides" ClassOrInterfaceType "with" { ClassOrInterfaceType { "," ClassOrInterfaceType } }
ModuleRequiresDirective ::= "requires" [ "transitive" ] [ "static" ] ModuleName
ModuleUsesDirective ::= "uses" ClassOrInterfaceType
ModuleOpensDirective ::= "opens" PackageName [ "to" { ModuleName { "," ModuleName } } ]
ModuleExportsDirective ::= "exports" PackageName [ "to" { ModuleName { "," ModuleName } } ]
UnionType ::= ReferenceType { "|" ReferenceType }
VarType ::= "var"
UnknownType ::= "unknown"
IntersectionType ::= ReferenceType { "&" ReferenceType }
JavadocComment ::= "/**" { Character } "*/"
LineComment ::= "//" { Character }
BlockComment ::= "/*" { Character } "*/"
EnumDeclaration ::= { Modifier } "enum" SimpleName "{" [ EnumConstantDeclaration { "," EnumConstantDeclaration } ] "}" [ Implements ]
AnnotationDeclaration ::= { Modifier } "@" "interface" SimpleName [ Extends ] "{" { AnnotationMemberDeclaration } "}"
ClassOrInterfaceDeclaration ::= { Modifier } [ "static" ] "class" SimpleName [ "extends" Type ] [ "implements" { Type { "," Type } } ]
RecordDeclaration ::= { Modifier } "record" SimpleName "(" { Parameter { "," Parameter } } ")" [ "extends" Type ] "{" { BodyDeclaration } "}" -->





