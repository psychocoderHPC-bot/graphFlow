/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "PhotonTypes.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace graphFlow::photon::diagnostics
{
    /** Environment and provenance fields written to `metadata.csv`. */
    struct RunMetadata
    {
        std::string command;
        std::string backend;
        std::string deviceName;
        std::string deviceApi;
        std::uint32_t deviceIndex = 0u;
        std::string graphflowCommit;
        std::string alpakaPin;
        std::string graphflowSourceDir;
        std::string alpakaSourceDir;
        std::uint32_t multiProcessorCount = 0u;
        std::uint32_t warpSize = 0u;
        std::uint64_t globalMemBytes = 0u;
    };

    inline char const* outcomeName(Event event)
    {
        switch(event)
        {
        case Event::transmitted:
            return "transmitted";
        case Event::reflected:
            return "reflected";
        case Event::absorbed:
            return "absorbed";
        case Event::capped:
            return "capped";
        case Event::scatter:
            return "scatter";
        }
        return "unknown";
    }

    inline char const* traceName(TraceKind kind)
    {
        switch(kind)
        {
        case TraceKind::source:
            return "source";
        case TraceKind::scatter:
            return "scatter";
        case TraceKind::transmitted:
            return "transmitted";
        case TraceKind::reflected:
            return "reflected";
        case TraceKind::absorbed:
            return "absorbed";
        case TraceKind::capped:
            return "capped";
        }
        return "unknown";
    }

    inline std::string csvQuote(std::string_view value)
    {
        std::string quoted{"\""};
        for(char const character : value)
        {
            if(character == '"')
                quoted += '"';
            quoted += character;
        }
        quoted += '"';
        return quoted;
    }

    inline std::string modeName(ScatterMode mode)
    {
        return mode == ScatterMode::forward ? "forward" : "isotropic";
    }

    /** Write the machine-readable run outputs. The caller creates no files itself. */
    inline void writeRunOutputs(
        std::filesystem::path const& outputDirectory,
        ScatteringResult const& result,
        Parameters const& parameters,
        RunMetadata const& metadata)
    {
        std::filesystem::create_directories(outputDirectory);
        std::ofstream metadataFile{outputDirectory / "metadata.csv"};
        std::ofstream outcomesFile{outputDirectory / "outcomes.csv"};
        std::ofstream histogramFile{outputDirectory / "scatter_histogram.csv"};
        std::ofstream exitsFile{outputDirectory / "exits.csv"};
        std::ofstream tracesFile{outputDirectory / "traces.csv"};
        std::ofstream traceSummaryFile{outputDirectory / "trace_summary.csv"};
        if(!metadataFile || !outcomesFile || !histogramFile || !exitsFile || !tracesFile || !traceSummaryFile)
            throw std::runtime_error{"graphFlow::photon: cannot create output files"};

        outcomesFile << "event,count\n";
        for(Event const event : {Event::transmitted, Event::reflected, Event::absorbed, Event::capped})
            outcomesFile << outcomeName(event) << ',' << result.eventCounts[static_cast<std::uint32_t>(event)] << '\n';

        histogramFile << "event,scatter_count,count\n";
        for(auto const& row : result.histograms)
            histogramFile << outcomeName(row.event) << ',' << row.scatterCount << ',' << row.count << '\n';

        exitsFile << "id,event,scatter_count,x,y,z,ux,uy,uz,cos_theta,phi_rad,transverse_x,transverse_y\n";
        exitsFile << std::setprecision(17);
        for(std::size_t id = 0u; id < result.terminal.size(); ++id)
        {
            TerminalRecord const& record = result.terminal[id];
            if(record.event != Event::transmitted && record.event != Event::reflected)
                continue;
            double const phi = std::atan2(record.direction.y, record.direction.x);
            exitsFile << id << ',' << outcomeName(record.event) << ',' << record.scatterCount << ','
                      << record.position.x << ',' << record.position.y << ',' << record.position.z << ','
                      << record.direction.x << ',' << record.direction.y << ',' << record.direction.z << ','
                      << record.direction.z << ',' << phi << ',' << record.position.x << ',' << record.position.y
                      << '\n';
        }

        tracesFile << "id,step,kind,scatter_count,x,y,z,ux,uy,uz\n";
        tracesFile << std::setprecision(17);
        traceSummaryFile << "id,vertices,truncated\n";
        for(auto const& trace : result.traces)
        {
            traceSummaryFile << trace.id << ',' << trace.count << ',' << (trace.truncated ? 1 : 0) << '\n';
            for(std::size_t step = 0u; step < trace.vertices.size(); ++step)
            {
                TraceVertex const& vertex = trace.vertices[step];
                tracesFile << trace.id << ',' << step << ',' << traceName(vertex.kind) << ',' << vertex.scatterCount
                           << ',' << vertex.position.x << ',' << vertex.position.y << ',' << vertex.position.z << ','
                           << vertex.direction.x << ',' << vertex.direction.y << ',' << vertex.direction.z << '\n';
            }
        }

        metadataFile << "key,value\n";
        auto const keyValue = [&](std::string_view key, std::string_view value)
        { metadataFile << key << ',' << csvQuote(value) << '\n'; };
        auto const keyNumber = [&](std::string_view key, auto value)
        { metadataFile << key << ',' << csvQuote(std::to_string(value)) << '\n'; };

        keyValue("schema_version", "1");
        keyValue("graphflow_commit", metadata.graphflowCommit);
        keyValue("alpaka_pin", metadata.alpakaPin);
        keyValue("graphflow_source_dir", metadata.graphflowSourceDir);
        keyValue("alpaka_source_dir", metadata.alpakaSourceDir);
        keyValue("backend", metadata.backend);
        keyNumber("device_index", metadata.deviceIndex);
        keyValue("device_name", metadata.deviceName);
        keyValue("device_api", metadata.deviceApi);
        keyNumber("multiprocessor_count", metadata.multiProcessorCount);
        keyNumber("warp_size", metadata.warpSize);
        keyNumber("global_mem_bytes", metadata.globalMemBytes);
        keyValue("command", metadata.command);
        keyValue("units", "length=1 coefficient=1/length angle=rad");
        keyNumber("photons", parameters.numPhotons);
        keyNumber("thickness", parameters.thickness);
        keyNumber("sigma_s", parameters.sigmaS);
        keyNumber("sigma_a", parameters.sigmaA);
        keyNumber("sigma_t", sigmaTotal(Slab{parameters.thickness, parameters.sigmaS, parameters.sigmaA}));
        keyValue("mode", modeName(parameters.mode));
        keyNumber("max_scatters", parameters.maxScatters);
        keyNumber("seed", parameters.seed);
        keyNumber("chunk_size", parameters.chunkSize);
        keyNumber("live_particles", result.liveParticles);
        keyNumber("tile_size", parameters.tileSize);
        keyNumber("batch_chunks", parameters.batchChunks);
        keyNumber("frame_cap", parameters.frameCap);
        keyNumber("pool_chunks", result.stats.poolChunks);
        keyNumber("peak_live_chunks", result.stats.peakLiveChunks);
        keyNumber("peak_reserved_chunks", result.stats.peakReservedChunks);
        keyNumber("count_terminated", result.terminated);
        keyNumber("count_inconsistent", result.inconsistent);
        keyNumber("count_transmitted", result.eventCounts[static_cast<std::uint32_t>(Event::transmitted)]);
        keyNumber("count_reflected", result.eventCounts[static_cast<std::uint32_t>(Event::reflected)]);
        keyNumber("count_absorbed", result.eventCounts[static_cast<std::uint32_t>(Event::absorbed)]);
        keyNumber("count_capped", result.eventCounts[static_cast<std::uint32_t>(Event::capped)]);
        keyNumber("trace_enabled", parameters.trace.enabled ? 1 : 0);
        keyNumber("trace_photons", parameters.trace.maxPhotons);
        keyNumber("trace_vertices", parameters.trace.maxVertices);
        keyNumber("trace_ids", result.traces.size());
        keyNumber("trace_truncated_ids", result.traceTruncated);
        keyNumber("detector_distance", parameters.thickness);
        keyNumber("detector_half_width", 2.0 * parameters.thickness);
    }
} // namespace graphFlow::photon::diagnostics
