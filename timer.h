#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

/**
 * Lightweight utility for timing code sections.
 *
 * Usage patterns in this project:
 *   Timer timer{"run_sandwich"};
 *   run_sandwich(...);
 *   timer.log_total_time();
 *
 * or measure a callable directly:
 *   auto [elapsed, result] = Timer::measure(run_sandwich, args...);
 */
class Timer {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Seconds = std::chrono::duration<double>;

    explicit Timer(std::string label = "Timer", bool start_now = true)
        : label_(std::move(label)), running_(false) {
        reset(start_now);
    }

    // Start or restart the timer; returns the previous elapsed time if it was running.
    Seconds start() {
        Seconds previous = running_ ? elapsed() : Seconds{0};
        running_ = true;
        start_time_ = Clock::now();
        last_lap_time_ = start_time_;
        stop_time_ = start_time_;
        return previous;
    }

    // Stop the timer and return total elapsed seconds.
    Seconds stop() {
        if (!running_) return stop_time_ - start_time_;
        stop_time_ = Clock::now();
        running_ = false;
        return stop_time_ - start_time_;
    }

    // Reset timer state; optionally restart immediately.
    void reset(bool restart = true) {
        start_time_ = Clock::now();
        last_lap_time_ = start_time_;
        stop_time_ = start_time_;
        running_ = restart;
    }

    // Record a lap and return the elapsed seconds since the previous lap.
    Seconds lap() {
        TimePoint now = running_ ? Clock::now() : stop_time_;
        Seconds delta = now - last_lap_time_;
        last_lap_time_ = now;
        return delta;
    }

    // Elapsed seconds since start (uses current time if still running).
    Seconds elapsed() const {
        TimePoint now = running_ ? Clock::now() : stop_time_;
        return now - start_time_;
    }

    // Convenience logging helpers -------------------------------------------------

    void log_lap(const std::string& message = "Lap",
                 std::ostream& os = std::cout) {
        Seconds delta = lap();
        os << "[Timer] " << label_ << " | " << message
           << " : " << delta.count() << " s\n";
    }

    void log_total_time(std::ostream& os = std::cout) const {
        os << "[Timer] " << label_ << " | total : "
           << elapsed().count() << " s\n";
    }

    // Static helpers --------------------------------------------------------------

    template <typename F, typename... Args>
    static auto measure(F&& func, Args&&... args) {
        Timer timer{"measure"};
        auto invoke = [&](){
            return std::invoke(std::forward<F>(func),
                               std::forward<Args>(args)...);
        };

        if constexpr (std::is_void_v<std::invoke_result_t<F, Args...>>){
            invoke();
            return timer.stop();
        } else {
            auto result = invoke();
            Seconds duration = timer.stop();
            return std::make_pair(duration, std::move(result));
        }
    }

private:
    std::string label_;
    TimePoint start_time_{};
    TimePoint last_lap_time_{};
    TimePoint stop_time_{};
    bool running_{false};
};

using TTimer = Timer;
using PTimer = std::shared_ptr<Timer>;
