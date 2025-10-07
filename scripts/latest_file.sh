latest_file() {
    latest=$(ls -v slurm_logs/*.out 2>/dev/null | tail -n 1)
    if [ -z "$latest" ]; then
        echo "No matching files found."
        return 1
    fi

    case "$1" in
        cat)
            cat "$latest"
            ;;
        watch)
            watch tail -n 10 "$latest"
            ;;
        tail)
            tail -n 10 "$latest"
            ;;
        nano)
            nano "$latest"
            ;;
        *)
            echo "Usage: latest_file [cat|watch|tail|nano]"
            return 1
            ;;
    esac
}