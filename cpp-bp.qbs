import qbs

Project {
    minimumQbsVersion: "1.7.1"

    CppApplication {
        consoleApplication: true
        cpp.cxxLanguageVersion: "c++17"

        files: [
            "main.cpp",
            "matrix.h",
            "perceptron.h",
        ]

        Group {     // Properties for the produced executable
            fileTagsFilter: "application"
            qbs.install: true
        }
    }
}
