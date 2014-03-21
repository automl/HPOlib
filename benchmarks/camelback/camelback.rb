#! /usr/bin/ruby

# ruby camelback.rb --fold 2 --folds 20 --params -x -0.0898 -y 0.7126
# >> Result for ParamILS: SAT, 0.000095, 1, -1.031628, -1, camelback.rb
# -2 < x < 2, -1 < y < 1
# 2 Global minima: f(x) = -1.0316 (0.0898, -0.7126), (-0.0898, 0.7126)

def camelback(x, y)
    puts 'Params: ', x, y

    tmp1 = (4 - 2.1 * x**2 + (x**4)/3) * x**2
    tmp2 = x*y
    tmp3 = (-4 + 4 * y**2) * y**2
    y = tmp1 + tmp2 + tmp3

    puts 'Result: ', y
    return y
end

starttime = Time.now()
x = -1
y = -1
puts ARGV
if ARGV[5] == '-y'
    y = ARGV[6].to_f
elsif ARGV[5] == '-x'
    x = ARGV[6].to_f
else

    abort("Result for ParamILS: CRASH, 1, 1, 10, -1, camelback.rb")
end

if ARGV[7] == '-y'
    y = ARGV[8].to_f
elsif ARGV[7] == '-x'
    x = ARGV[8].to_f
else
    abort("Result for ParamILS: CRASH, 1, 1, 10, -1, camelback.rb")
end

result = camelback(x, y)

# Benchmark need to take some time, because of a bug in runsolver
sleep(1)
duration = Time.now() - starttime
printf "\nResult for ParamILS: %s, %f, 1, %f, %d, %s\n", "SAT", duration, result, -1, "camelback.rb"