cmake_minimum_required(VERSION 3.2)

project(Example1)

add_executable(Example1 example1.cpp)
add_executable(Example2 example2.cpp)
add_executable(Example3 example3.cpp)
if(USE_OPENMP)
	add_executable(Example4 example4.cpp)
endif(USE_OPENMP)

add_executable(Example5 example5.cpp)
add_executable(Example7 example7.cpp)
add_executable(ExampleNeohookeanTet exampleNeohookeanTet.cpp)

add_executable(ExampleEigenFit exampleEigenFit.cpp)
target_compile_options(ExampleEigenFit PUBLIC -DNH)


add_executable(ExampleEigenFitCoRot exampleEigenFit.cpp)
target_compile_options(ExampleEigenFitCoRot PUBLIC -DCOROT)


add_executable(ExampleEigenFitLinear exampleEigenFit.cpp)
target_compile_options(ExampleEigenFitLinear PUBLIC -DLINEAR)

add_executable(ExampleEigenFitNH exampleEigenFit.cpp)
target_compile_options(ExampleEigenFitNH PUBLIC -DNH)

add_executable(ExampleEigenFitARAP exampleEigenFit.cpp)
target_compile_options(ExampleEigenFitARAP PUBLIC -DARAP)





add_executable(ExampleEigenFitNoExit exampleEigenFit.cpp)
target_compile_options(ExampleEigenFit PUBLIC -DNH -DNOEXIT)


add_executable(ExampleEigenFitCoRotNoExit exampleEigenFit.cpp)
target_compile_options(ExampleEigenFitCoRot PUBLIC -DCOROT -DNOEXIT)


add_executable(ExampleEigenFitLinearNoExit exampleEigenFit.cpp)
target_compile_options(ExampleEigenFitLinear PUBLIC -DLINEAR -DNOEXIT)

add_executable(ExampleEigenFitNHNoExit exampleEigenFit.cpp)
target_compile_options(ExampleEigenFitNH PUBLIC -DNH -DNOEXIT)

add_executable(ExampleEigenFitARAPNoExit exampleEigenFit.cpp)
target_compile_options(ExampleEigenFitARAP PUBLIC -DARAP -DNOEXIT)






#add_executable(ExampleEigenFitEXP exampleEigenFit.cpp)
#target_compile_options(ExampleEigenFitEXP PUBLIC -DNH)
#target_compile_options(ExampleEigenFitEXP PUBLIC -DEXP)

#add_executable(ExampleEigenFitCoRotEXP exampleEigenFit.cpp)
#target_compile_options(ExampleEigenFitCoRotEXP PUBLIC -DCOROT)

#add_executable(ExampleEigenFitARAPEXP exampleEigenFit.cpp)
#target_compile_options(ExampleEigenFitARAPEXP PUBLIC -DARAP)
#target_compile_options(ExampleEigenFitARAPEXP PUBLIC -DEXP)



add_executable(ExampleExponentialIMEX exampleExponentialIMEX.cpp)
target_compile_options(ExampleExponentialIMEX PUBLIC -DNH)

add_executable(ExampleExponentialIMEXCoRot exampleExponentialIMEX.cpp)
target_compile_options(ExampleExponentialIMEXCoRot PUBLIC -DCOROT)

add_executable(ExampleExponentialIMEXLinear exampleExponentialIMEX.cpp)
target_compile_options(ExampleExponentialIMEXLinear PUBLIC -DLINEAR)

add_executable(ExampleExponentialIMEXNH exampleExponentialIMEX.cpp)
target_compile_options(ExampleExponentialIMEXNH PUBLIC -DNH)

add_executable(ExampleExponentialIMEXARAP exampleExponentialIMEX.cpp)
target_compile_options(ExampleExponentialIMEXARAP PUBLIC -DARAP)

add_executable(ExampleExponentialIMEXSTVK exampleExponentialIMEX.cpp)
target_compile_options(ExampleExponentialIMEXSTVK PUBLIC -DSTVK)


#add_executable(ExampleExponentialIMEX exampleExponentialIMEX.cpp)
#add_executable(ExampleERE exampleERE.cpp)
#add_executable(ExampleSI exampleSI.cpp)
#add_executable(ExampleBE exampleBE.cpp)
#add_executable(ExampleIM exampleIM.cpp)
add_executable(Example10 example10.cpp)


add_executable(ExampleCollisionsFloor exampleCollisionsFloor.cpp)
target_link_libraries(ExampleCollisionsFloor Core Base Collisions FEM UI ${GAUSS_LIBS})


add_executable(ExampleImplicitIntegration exampleImplicitIntegration.cpp)
target_link_libraries(ExampleImplicitIntegration Core Base FEM Optimization UI ${GAUSS_LIBS})


add_executable(Example13 example13.cpp)
add_executable(Example14 example14.cpp)
add_executable(Example15 example15.cpp)
add_executable(Example16 example16.cpp)
add_executable(Example17 example17.cpp)
add_executable(Example18 example18.cpp)
add_executable(ExampleTestFCL exampleTestFCL.cpp)
add_executable(ExampleTwistingBar exampleTwistingBar.cpp)
add_executable(ExampleRigidBody exampleRigidBody.cpp)
add_executable(ExampleNewmarkNonlinear exampleNewmarkNonlinear.cpp)
add_executable(ExamplePrincipalStretch examplePrincipalStretch.cpp)
add_executable(ExampleLoubignac exampleLoubignac.cpp)

#add_executable(ExampleFloatingFrame exampleFloatingFrame.cpp)

target_link_libraries(Example1 Core Base UI ${GAUSS_LIBS})
target_link_libraries(Example2 Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(Example3 Core Base FEM UI ${GAUSS_LIBS})

if(USE_OPENMP)
	target_link_libraries(Example4 Core Base FEM UI ${GAUSS_LIBS})
endif(USE_OPENMP)

add_executable(ExampleLinearModes exampleLinearModes.cpp)
target_link_libraries(ExampleLinearModes Core Base FEM UI ${GAUSS_LIBS})

target_link_libraries(Example5 Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(Example7 Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleNeohookeanTet Core Base FEM UI ${GAUSS_LIBS})

target_link_libraries(ExampleEigenFit Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleEigenFitCoRot Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleEigenFitARAP Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleEigenFitNH Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleEigenFitLinear Core Base FEM UI ${GAUSS_LIBS})

target_link_libraries(ExampleEigenFitNoExit Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleEigenFitCoRotNoExit Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleEigenFitARAPNoExit Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleEigenFitNHNoExit Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleEigenFitLinearNoExit Core Base FEM UI ${GAUSS_LIBS})


#target_link_libraries(ExampleEigenFitEXP Core Base FEM UI ${GAUSS_LIBS})
#target_link_libraries(ExampleEigenFitCoRotEXP Core Base FEM UI ${GAUSS_LIBS})
#target_link_libraries(ExampleEigenFitARAPEXP Core Base FEM UI ${GAUSS_LIBS})

target_link_libraries(ExampleExponentialIMEX Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleExponentialIMEXLinear Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleExponentialIMEXCoRot Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleExponentialIMEXARAP Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleExponentialIMEXNH Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleExponentialIMEXSTVK Core Base FEM UI ${GAUSS_LIBS})

#target_link_libraries(ExampleERE Core Base FEM UI ${GAUSS_LIBS})
#target_link_libraries(ExampleSI Core Base FEM UI ${GAUSS_LIBS})
#target_link_libraries(ExampleBE Core Base FEM UI ${GAUSS_LIBS})
#target_link_libraries(ExampleIM Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(Example10 Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(Example13 Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(Example14 Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(Example15 Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(Example16 Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(Example17 Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(Example18 Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleTestFCL Core Base FEM fcl ccd UI ${GAUSS_LIBS})
target_link_libraries(ExampleTwistingBar Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleRigidBody Core Base RigidBodies UI ${GAUSS_LIBS})
target_link_libraries(ExampleNewmarkNonlinear Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExamplePrincipalStretch Core Base FEM UI ${GAUSS_LIBS})
target_link_libraries(ExampleLoubignac igl::core igl::opengl_glfw igl::opengl_glfw_imgui Core Base FEM UI ${GAUSS_LIBS})

#target_link_libraries(ExampleFloatingFrame Core Base RigidBodies UI ${GAUSS_LIBS})
