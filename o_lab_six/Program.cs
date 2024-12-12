using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace LagrangeMultipliersExample
{
    class Program
    {
        static void Main(string[] args)
        {
            // Начальное приближение для переменных: x1, x2, lambda1, lambda2, lambda3, mu1
            double[] initialGuess = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

            // Максимальное количество итераций и точность
            int maxIterations = 100;
            double tolerance = 1e-6;

            // Решаем систему уравнений методом Ньютона
            bool success = NewtonRaphsonSolver(initialGuess, out double[] solution, maxIterations, tolerance);

            if (success)
            {
                Console.WriteLine("Решение найдено:");
                Console.WriteLine($"x1 = {solution[0]}");
                Console.WriteLine($"x2 = {solution[1]}");
                Console.WriteLine($"lambda1 = {solution[2]}");
                Console.WriteLine($"lambda2 = {solution[3]}");
                Console.WriteLine($"lambda3 = {solution[4]}");
                Console.WriteLine($"mu1 = {solution[5]}");
                Console.WriteLine($"Целевая функция f(x*) = {3 * solution[0] + 4 * solution[1]}");

                // Проверка условий комплементарности
                Console.WriteLine("\nПроверка условий комплементарности:");
                Console.WriteLine($"lambda1 * g1(x) = {solution[2] * (solution[0] + solution[1] - 4)}");
                Console.WriteLine($"lambda2 * g2(x) = {solution[3] * (-solution[0])}");
                Console.WriteLine($"lambda3 * g3(x) = {solution[4] * (-solution[1])}");
            }
            else
            {
                Console.WriteLine("Не удалось найти решение.");
            }
        }

        /// <summary>
        /// Метод для решения системы нелинейных уравнений методом Ньютона-Рафсона.
        /// </summary>
        /// <param name="initialGuess">Начальное приближение для переменных.</param>
        /// <param name="solution">Найденное решение.</param>
        /// <param name="maxIterations">Максимальное количество итераций.</param>
        /// <param name="tolerance">Точность решения.</param>
        /// <returns>Успешность нахождения решения.</returns>
        static bool NewtonRaphsonSolver(double[] initialGuess, out double[] solution, int maxIterations, double tolerance)
        {
            // Инициализация решения
            solution = (double[])initialGuess.Clone();
            int n = solution.Length;

            for (int iter = 0; iter < maxIterations; iter++)
            {
                // Вычисляем значения системы уравнений в текущей точке
                double[] F = ComputeSystem(solution);

                // Проверяем, достаточно ли мал нормы F
                double normF = 0.0;
                foreach (double fi in F)
                {
                    normF += fi * fi;
                }
                normF = Math.Sqrt(normF);
                if (normF < tolerance)
                {
                    return true;
                }

                // Вычисляем Якобиан
                double[,] J = ComputeJacobian(solution);
                var jacobianMatrix = Matrix<double>.Build.DenseOfArray(J);

                // Решаем систему J * delta = -F
                var FVector = Vector<double>.Build.Dense(F);
                var delta = jacobianMatrix.Solve(-FVector);

                // Обновляем решение
                for (int i = 0; i < n; i++)
                {
                    solution[i] += delta[i];
                }

                // Проверяем, не превысило ли решение допустимые пределы (например, неотрицательность)
                for (int i = 0; i < n; i++)
                {
                    if (solution[i] < 0)
                    {
                        solution[i] = 0;
                    }
                }
            }

            // Если достигнуто максимальное количество итераций
            return false;
        }

        /// <summary>
        /// Вычисляет значения системы уравнений в заданной точке.
        /// </summary>
        /// <param name="vars">Массив переменных: x1, x2, lambda1, lambda2, lambda3, mu1.</param>
        /// <returns>Массив значений системы уравнений.</returns>
        static double[] ComputeSystem(double[] vars)
        {
            double x1 = vars[0];
            double x2 = vars[1];
            double lambda1 = vars[2];
            double lambda2 = vars[3];
            double lambda3 = vars[4];
            double mu1 = vars[5];

            double[] F = new double[6];

            // Уравнение 1: ∂L/∂x1 = 3 + lambda1 - lambda2 + 2 * mu1 * x1 = 0
            F[0] = 3 + lambda1 - lambda2 + 2 * mu1 * x1;

            // Уравнение 2: ∂L/∂x2 = 4 + lambda1 - lambda3 + 2 * mu1 * x2 = 0
            F[1] = 4 + lambda1 - lambda3 + 2 * mu1 * x2;

            // Уравнение 3: ∂L/∂lambda1 = x1 + x2 - 4 = 0
            F[2] = x1 + x2 - 4;

            // Уравнение 4: ∂L/∂lambda2 = -x1 = 0
            F[3] = -x1;

            // Уравнение 5: ∂L/∂lambda3 = -x2 = 0
            F[4] = -x2;

            // Уравнение 6: ∂L/∂mu1 = x1^2 + x2^2 - 4 = 0
            F[5] = x1 * x1 + x2 * x2 - 4;

            return F;
        }

        /// <summary>
        /// Вычисляет матрицу Якоби системы уравнений в заданной точке.
        /// </summary>
        /// <param name="vars">Массив переменных: x1, x2, lambda1, lambda2, lambda3, mu1.</param>
        /// <returns>Матрица Якоби.</returns>
        static double[,] ComputeJacobian(double[] vars)
        {
            double x1 = vars[0];
            double x2 = vars[1];
            double lambda1 = vars[2];
            double lambda2 = vars[3];
            double lambda3 = vars[4];
            double mu1 = vars[5];

            double[,] J = new double[6, 6];

            // Частные производные по x1
            J[0, 0] = 2 * mu1;          // dF1/dx1
            J[0, 1] = 0;                // dF1/dx2
            J[0, 2] = 1;                // dF1/dlambda1
            J[0, 3] = -1;               // dF1/dlambda2
            J[0, 4] = 0;                // dF1/dlambda3
            J[0, 5] = 2 * x1;           // dF1/dmu1

            // Частные производные по x2
            J[1, 0] = 0;                // dF2/dx1
            J[1, 1] = 2 * mu1;          // dF2/dx2
            J[1, 2] = 1;                // dF2/dlambda1
            J[1, 3] = 0;                // dF2/dlambda2
            J[1, 4] = -1;               // dF2/dlambda3
            J[1, 5] = 2 * x2;           // dF2/dmu1

            // Частные производные по lambda1
            J[2, 0] = 1;                // dF3/dx1
            J[2, 1] = 1;                // dF3/dx2
            J[2, 2] = 0;                // dF3/dlambda1
            J[2, 3] = 0;                // dF3/dlambda2
            J[2, 4] = 0;                // dF3/dlambda3
            J[2, 5] = 0;                // dF3/dmu1

            // Частные производные по lambda2
            J[3, 0] = -1;               // dF4/dx1
            J[3, 1] = 0;                // dF4/dx2
            J[3, 2] = 0;                // dF4/dlambda1
            J[3, 3] = 0;                // dF4/dlambda2
            J[3, 4] = 0;                // dF4/dlambda3
            J[3, 5] = 0;                // dF4/dmu1

            // Частные производные по lambda3
            J[4, 0] = 0;                // dF5/dx1
            J[4, 1] = -1;               // dF5/dx2
            J[4, 2] = 0;                // dF5/dlambda1
            J[4, 3] = 0;                // dF5/dlambda2
            J[4, 4] = 0;                // dF5/dlambda3
            J[4, 5] = 0;                // dF5/dmu1

            // Частные производные по mu1
            J[5, 0] = 2 * x1;           // dF6/dx1
            J[5, 1] = 2 * x2;           // dF6/dx2
            J[5, 2] = 0;                // dF6/dlambda1
            J[5, 3] = 0;                // dF6/dlambda2
            J[5, 4] = 0;                // dF6/dlambda3
            J[5, 5] = 0;                // dF6/dmu1

            return J;
        }
    }
}
