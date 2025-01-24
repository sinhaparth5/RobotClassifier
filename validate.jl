using Images, FileIO

function check_images(path)
    bad_files = String[]
    for (root, dirs, files) in walkdir(path)
        for file in files
            full_path = joinpath(root, file)
            try
                load(full_path)
            catch e
                push!(bad_files, full_path)
                @warn "Corrupted file: $full_path"
                rm(full_path; force=true)
            end
        end
    end
    return bad_files
end

check_images("data")